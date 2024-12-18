import math
import copy

import torch
import numpy as np
import mmengine
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

from dreamer_datasets import DefaultCollator, DefaultSampler, load_dataset, utils, ImageVisualizer, image_utils
GLIGEN_WEIGHT_NAME = 'pytorch_gligen_weights.bin'


def choose_objs(boxes, labels, max_objs, label_names=None, shuffle=False):
    if len(boxes) == 0:
        return []
    if label_names is not None:
        keeps = []
        for i in range(len(labels)):
            if labels[i] in label_names:
                keeps.append(i)
        keeps = np.array(keeps, dtype=np.int64)
    else:
        keeps = np.arange(len(labels))
    if len(keeps) > max_objs:
        areas = []
        for i in keeps:
            x1, y1, x2, y2 = boxes[i]
            areas.append((x2 - x1) * (y2 - y1))
        indexes = np.array(areas).argsort()[::-1]
        keeps = keeps[indexes[:max_objs]]
    if shuffle:
        keeps = np.random.permutation(keeps)
    return keeps


def pad_data(data, max_objs):
    data_shape = list(data.shape)
    assert data_shape[0] <= max_objs
    data_shape[0] = max_objs
    new_data = np.zeros(data_shape, dtype=data.dtype)
    new_data[: len(data)] = data
    return new_data

class VideoSampler(DefaultSampler):
    def __init__(self, dataset, data_idx_info, batch_size=None, cam_num=6, frame_num=32, hz_factor=1, mv_video=False,cam_names=None ,view='CAM_FRONT', shuffle=False, resample=False, infinite=True, seed=6666,front_condition=False):
        super(VideoSampler, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, infinite=infinite, seed=seed)
        self.view = view
        self.mv_video = mv_video
        self.hz_factor = hz_factor
        self.data_idx_info = mmengine.load(data_idx_info)
        # self.cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

        self.cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'] if cam_names is None else cam_names
        
        self.cam_num = cam_num
        self.frame_num = frame_num
        self.batch_size = batch_size
        self.resample = resample
        
        bs_fac = int(batch_size / self.cam_num / self.frame_num)
        # multiview init frames: [FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, ....]
        if mv_video:
            front_indexes = [i for i in range(len(self.data_idx_info)) if self.data_idx_info[i]['CAM_NAME']=='CAM_FRONT' and self.data_idx_info[i]['start']]
            self.index = []
            if front_condition:
                for front_idx in front_indexes:
                    this_idx = [self.data_idx_info[front_idx]['multiview_start_idx'][cam_name] for cam_name in self.cam_names]
                    this_idx.insert(0, front_idx)
                    self.index.extend(this_idx)
            elif len(self.cam_names)==5:
                for front_idx in front_indexes:
                    this_idx = [self.data_idx_info[front_idx]['multiview_start_idx'][cam_name] for cam_name in self.cam_names]
                    this_idx.insert(1, front_idx)
                    self.index.extend(this_idx)
            elif len(self.cam_names) ==2:
                for front_idx in front_indexes:
                    this_idx = [self.data_idx_info[front_idx]['multiview_start_idx'][cam_name] for cam_name in self.cam_names]
                    this_idx.insert(0, front_idx)
                    self.index.extend(this_idx)
            else:
                assert False
             
        # single-view init frames: [C1, C2, C3, xxx]
        else:
            if view != 'ALL':
                self.index = [i for i in range(len(self.data_idx_info)) if self.data_idx_info[i]['CAM_NAME']==view and self.data_idx_info[i]['start']]
            else:
                self.index = [i for i in range(len(self.data_idx_info)) if self.data_idx_info[i]['start']]
                
        if self.resample:
            # TODO resample for mv videos
            assert view != 'ALL' and not self.mv_video
            print('resampling data based on action input.... make sure you need this operation.')
            # 1. statistically analyze yaw mean
            yaws = []
            for idx in self.index:
                yaw_ = [np.abs(self.data_idx_info[idx+j*self.hz_factor]['state']['yaw']) for j in range(self.frame_num)]
                yaws.append(np.mean(yaw_))
            # 2. resmaple data based on yaw mean
            new_idx = []
            yaws = np.array(yaws)
            yaw_bias = 0.02  # rad bias  TODO hardcode
            resample_log_fac = 0.2
            for i, idx in enumerate(self.index):
                yaw = yaws[i]
                yaw_bias_num = ((yaws>yaw-yaw_bias) & (yaws<yaw+yaw_bias)).sum()
                yaw_prob = yaw_bias_num / len(yaws)
                yaw_sample_fac = np.ceil(np.log(resample_log_fac/(yaw_prob+1e-5) + 1))
                new_idx.extend([idx for j in range(int(yaw_sample_fac))])
            print('before resample, data size: ', len(self.index))
            self.index = new_idx
            print('after resample, data size: ', len(self.index))
            
        if self.mv_video:
            self.total_size = int(math.ceil(len(self.index) / self.cam_num / bs_fac)) * bs_fac * self.frame_num * self.cam_num
        else:
            self.total_size = math.ceil(len(self.index) / bs_fac) * self.frame_num * bs_fac
            
    def __iter__(self):
        total_size = int(self.total_size/self.frame_num)
            
        while True:
            indices = self.index
            while len(indices) < total_size:
                indices_i = copy.deepcopy(indices)
                num_data = min(len(indices_i), total_size - len(indices))
                indices = np.hstack((indices, indices_i[:num_data]))
                
            if self.shuffle:
                # multiview init frames: [FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, ....]
                if self.mv_video:
                    indices = np.array(indices)
                    indices = indices.reshape(-1, self.cam_num)
                    indices = indices.tolist()
                    indices = np.random.permutation(indices)
                    indices = indices.reshape(-1).tolist()
                    
                # single-view init frames: [C1, C2, C3, xxx]
                else:
                    indices = np.random.permutation(indices)
            
            """ note bellow is ok for multi viewo, and the format is 
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            ...
            """
            new_indices = np.stack([np.array(indices) + i*self.hz_factor for i in range(self.frame_num)]).T.reshape(-1)

            yield from new_indices
            if not self.infinite:
                break

class VideoCollator(DefaultCollator):
    def __init__(self, frame_num, img_mask_type, img_mask_num):
      self.frame_num = frame_num
      self.img_mask_type = img_mask_type
      self.img_mask_num = img_mask_num
            
    def __call__(self, batch):
        batch_dict = dict()
        if isinstance(batch, list):
            for key in batch[0]:
                batch_dict[key] = self._collate([d[key] for d in batch])
        elif isinstance(batch, dict):
            for key in batch:
                batch_dict[key] = self._collate(batch[key])
        else:
            assert False
        
        # process image conditions
        if self.img_mask_type is not None:
            assert 'input_image' in batch_dict
            img_cond_mask = torch.zeros_like(batch_dict['input_image'])
            
            if self.img_mask_type == 'prediction':
                img_cond_idx = np.arange(self.img_mask_num)
            elif self.img_mask_type == 'interpolation':
                assert self.img_mask_num > 1
                img_cond_idx = np.arange(0, self.frame_num, self.frame_num // (self.img_mask_num - 1))
            elif self.img_mask_type == 'random':
                img_cond_idx = np.random.choice(self.frame_num, self.img_mask_num, False)
            else:
                raise NotImplementedError
            img_cond_mask[img_cond_idx] = 1
            img_cond = torch.concat([
                batch_dict['input_image'] * img_cond_mask,
                img_cond_mask[:,0:1]], dim=1)
            batch_dict['img_cond_downsampler_input'] = img_cond

        return batch_dict



def draw_mv_video(videos, batch_dict):
    output_images = []
    gen_images1 = videos
    width,height = gen_images1[0].size
    width = width//6
    img_size = (width,height) 
    frame_num = len(gen_images1)
    imgs = []
    boxes = []
    hdmaps = []
    for j in range(6):
        for i in range(len(batch_dict['image'])//48):
            if i<len(batch_dict['image'])//48-1:
                video_len=7
            else:
                video_len = 8
            img = batch_dict['image'][i*48+j*8:i*48+j*8+video_len]
            box = batch_dict['image_box'][i*48+j*8:i*48+j*8+video_len]
            hdmap = batch_dict['image_hdmap'][i*48+j*8:i*48+j*8+video_len]
            imgs+=img
            boxes+=box
            hdmaps+=hdmap
    for i in range(frame_num):
        ori_img = []
        ann_img = []
        for j in range(6):
            ori_img.append(imgs[i+frame_num*j])#.resize(img_size))
            image = ImageVisualizer(np.zeros_like(np.array(ori_img[0])))
            if 'image_hdmap' in batch_dict:
                image_hdmap = hdmaps[i+j*frame_num]
                image_hdmap = image_hdmap#.resize(img_size)
                image_hdmap = np.array(image_hdmap)[:, :, ::-1]
                image.draw_seg(image_hdmap, scale=1.0)
           
                canvas_box=boxes[i+j*frame_num]
                if canvas_box.device != 'cpu':
                    canvas_box = canvas_box.cpu()
                canvas_box = canvas_box.sum(axis=0)
                image_box = Image.fromarray(canvas_box.numpy()).resize(img_size)
                image_box = np.array(image_box)
                image.add_boxes(image_box)
            ann_img.append(image.get_image())
        gt_img = image_utils.concat_images(ori_img, pad=0)
        canvas_img = image_utils.concat_images(ann_img, pad=0)
        this_img = image_utils.concat_images([gt_img,canvas_img,gen_images1[i]],direction='vertical',pad=2)
        output_images.append(this_img)
   
    
    return output_images

def draw_mv_video_v2(videos, batch_dict):
    output_images = []
    gen_images1 = videos[0]
    gen_images2 = videos[1]
    gen_images3 = videos[2]
    width,height = gen_images1[0].size
    width = width//6
    img_size = (width,height) 
    frame_num = len(gen_images1)
    for i in range(frame_num):
        ori_img = []
        ann_img = []
        for j in range(6):
            ori_img.append(batch_dict['image'][i+frame_num*j].resize(img_size))
            image = ImageVisualizer(np.zeros_like(np.array(ori_img[0])))
            if 'image_hdmap' in batch_dict:
                image_hdmap = batch_dict['image_hdmap'][i+j*frame_num]
                image_hdmap = image_hdmap.resize(img_size)
                image_hdmap = np.array(image_hdmap)[:, :, ::-1]
                image.draw_seg(image_hdmap, scale=1.0)
            canvas_box = batch_dict.get('image_box',None)
            if canvas_box is not None:
                canvas_box=canvas_box[i+j*frame_num]
                if canvas_box.device != 'cpu':
                    canvas_box = canvas_box.cpu()
                canvas_box = canvas_box.sum(axis=0)
                image_box = Image.fromarray(canvas_box.numpy()).resize(img_size)
                image_box = np.array(image_box)
                image.add_boxes(image_box)
            ann_img.append(image.get_image())
        gt_img = image_utils.concat_images(ori_img, pad=0)
        canvas_img = image_utils.concat_images(ann_img, pad=0)
        this_img = image_utils.concat_images([gt_img,canvas_img,gen_images1[i],gen_images2[i],gen_images3[i]],direction='vertical',pad=2)
        output_images.append(this_img)
   
    
    return output_images

