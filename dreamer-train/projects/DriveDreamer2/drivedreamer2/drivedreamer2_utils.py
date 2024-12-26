import math
import copy

import torch
import numpy as np
import mmengine
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from dreamer_datasets import DefaultCollator, DefaultSampler, load_dataset, utils, ImageVisualizer, image_utils
GLIGEN_WEIGHT_NAME = 'pytorch_gligen_weights.bin'


def custom_collate_fn(batch):
    frame_idx = [item['frame_idx'] for item in batch]
    cam_type = [item['cam_type'] for item in batch]
    video_length = [item['video_length'] for item in batch]
    multiview_start_idx = [item['multiview_start_idx'] if 'multiview_start_idx' in item else None for item in batch]
    return {
        'frame_idx': frame_idx,
        'cam_type': cam_type,
        'video_length': video_length,
        'multiview_start_idx': multiview_start_idx
    }

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

class VideoSampler(DefaultSampler):
    def __init__(self, dataset,batch_size=None, cam_num=6, frame_num=32, hz_factor=1, video_split_rate=2, mv_video=False, view='cam_front', shuffle=False, infinite=True, seed=6666, logger=None, 
                 resample_num_workers=8, resample_batch_size=64):
        super(VideoSampler, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, infinite=infinite, seed=seed)
        self.view = view
        self.mv_video = mv_video
        self.hz_factor = hz_factor
        # self.cam_names = ['cam_front_left', 'cam_front_right', 'cam_back_right', 'cam_back', 'cam_back_left']
        self.cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        self.cam_num = cam_num
        self.frame_num = frame_num
        self.data_num_per_batch = cam_num * frame_num
        self.img_batch_size = batch_size
        video_batch_size = int(self.img_batch_size / self.data_num_per_batch)
        
        # process index according to frame_num
        logger.info('Sampling video data from image dataset (depends on num_frames, hz_factor, video_split_rate), this may take minutes...')
        logger.info('For faster debugging, please use mini-version of nuscene.')
        dataloader = DataLoader(dataset, batch_size=resample_batch_size, num_workers=resample_num_workers, collate_fn=custom_collate_fn)
        video_frame_len = hz_factor * frame_num
        video_first_frame_flag = []
        video_front_view_idxes = []
        multiview_start_idxes = []
        offset_idx = 0
        
        for batch in tqdm(dataloader):
            frame_idxs = np.array(batch['frame_idx'])
            video_lengths = np.array(batch['video_length'])
            flags = (frame_idxs % (video_frame_len // video_split_rate) == 0) & (frame_idxs + video_frame_len <= video_lengths)
            video_first_frame_flag.extend(flags.tolist())
            front_idx = np.where(np.array(batch['cam_type']) == 'cam_front')[0]
            video_front_view_idxes.extend((offset_idx + front_idx).tolist())
            offset_idx += len(batch['frame_idx'])
            if mv_video:
                multiview_start_idxes.extend(batch['multiview_start_idx'])
                    
        # multiview frames: [FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, ....]
        if mv_video:
            front_indexes = [i for i in video_front_view_idxes if video_first_frame_flag[i]]
            self.index = []
            for front_idx in front_indexes:
                this_idx = [multiview_start_idxes[front_idx][cam_name] for cam_name in self.cam_names]
                this_idx.insert(1, front_idx)
                self.index.extend(this_idx)
                
        # single-view frames: [C1, C2, C3, ...]
        else:
            if view != 'ALL':
                self.index = [i for i in video_front_view_idxes if video_first_frame_flag[i]]
            else:
                self.index = [i for i in range(len(dataset)) if video_first_frame_flag[i]]
            
        if self.mv_video:
            self.total_size = int(math.ceil(len(self.index) / self.cam_num / video_batch_size)) * video_batch_size * self.data_num_per_batch
        else:
            self.total_size = math.ceil(len(self.index) / video_batch_size) * self.frame_num * video_batch_size
            
        logger.info('Done sampling!')
            
    def __iter__(self):
        video_size = int(self.total_size/self.frame_num)
            
        while True:
            indices = self.index
            while len(indices) < video_size:
                indices_i = copy.deepcopy(indices)
                num_data = min(len(indices_i), video_size - len(indices))
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
            
            """ note bellow is ok for multi video, and the format is 
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            ...
            """
            new_indices = np.stack([np.array(indices) + i*self.hz_factor for i in range(self.frame_num)]).T.reshape(-1)

            yield from new_indices
            if not self.infinite:
                break

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

