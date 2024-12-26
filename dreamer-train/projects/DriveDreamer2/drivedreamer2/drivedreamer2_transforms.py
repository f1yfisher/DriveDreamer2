import random
import numpy as np
from torchvision import transforms  
from .drivedreamer2_utils import choose_objs,pad_data
from dreamer_datasets import boxes3d_utils
import mmengine
import cv2

HEIGHT,WIDTH=900,1600

class DriveDreamer2_Transform:
    def __init__(
        self,
        resolution,
        gd_input_name=None,
        pos_input_name=None,
        bd_input_name=None,
        is_train=False,
        embed_map_path = '/data/disk2/public/datasets/driveDreamerV2/nuscenes/clip_text_transform.pkl',
        text_embed_dim = 768,
        box_normal=False,
    ):
        self.resolution = resolution
        self.gd_input_name = gd_input_name
        self.pos_input_name = pos_input_name
        self.bd_input_name = bd_input_name
        self.is_train = is_train
        self.resolution=resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.text_embed_dim = text_embed_dim
        self.box_normal = box_normal
        self.init_map(embed_map_path)
        self.box_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def init_map(self,embed_map_path):
        self.name_map = {
            'animal': 0,
            'human.pedestrian.adult': 1,
            'human.pedestrian.child': 2,
            'human.pedestrian.construction_worker': 3,
            'human.pedestrian.police_officer': 4,
            'human.pedestrian.personal_mobility': None,
            'human.pedestrian.stroller': None,
            'human.pedestrian.wheelchair': None,
            'vehicle.bicycle': 5,
            'vehicle.motorcycle': 6,
            'static_object.bicycle_rack': 7,
            'vehicle.car': 8,
            'vehicle.truck': 9,
            'vehicle.bus.bendy': 10,
            'vehicle.bus.rigid': 11,
            'vehicle.construction': 12,
            'vehicle.emergency.ambulance': 13,
            'vehicle.emergency.police': 14,
            'vehicle.trailer': 16,
            'movable_object.barrier': 17,
            'movable_object.trafficcone': 18,
            'movable_object.debris': None,
            'movable_object.pushable_pullable': None,
            }
        self.box_skeleton=[[0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]]
        _, self.prompt_embed_map = mmengine.load(embed_map_path)

    def generate_canvas_box(
        self,
        corners,
        idxs,
        thickness=2,
        data_idx=None,
        debug = False
    ):
        def pt_out_img(pt):
            return pt[0]<0 or pt[0]>=WIDTH or pt[1]<0 or pt[1]>=HEIGHT
        canvas_box=np.zeros((19,HEIGHT,WIDTH))
        # corners: (num_boxes, 8, 2)
        if len(corners) == 0:
            return canvas_box
        corners = np.array(corners, dtype=np.int32)
        corners = corners.reshape((corners.shape[0], 8, 2))
        if self.box_normal:
            color_line = 1
            color_face = 0.8
        else:
            color_line = 255
            color_face = 200
        for i, corner in enumerate(corners):
            idx = idxs[i]
            w_1 = abs(corner[0,0]-corner[2,0])
            h_1 = abs(corner[0,1]-corner[2,1])
            w_2 = abs(corner[4,0]-corner[6,0])
            h_2 = abs(corner[4,1]-corner[6,1])
            if w_1*h_1<w_2*h_2:
                for i_st,i_end in self.box_skeleton:
                    if pt_out_img(corner[i_st]) and pt_out_img(corner[i_end]):
                        continue
                    cv2.line(
                        canvas_box[idx],
                        (corner[i_st, 0], corner[i_st, 1]),
                        (corner[i_end, 0], corner[i_end, 1]),
                        color =color_line,
                        thickness=thickness,
                    )
                if not (pt_out_img(corner[4])and pt_out_img(corner[5]) \
                  and pt_out_img(corner[6]) and pt_out_img(corner[7])):
                    cv2.fillPoly(
                          canvas_box[idx], 
                          [np.array([corner[4],corner[5],corner[6],corner[7]])],
                          color=color_face)
            else:
                if not (pt_out_img(corner[4])and pt_out_img(corner[5]) \
                  and pt_out_img(corner[6]) and pt_out_img(corner[7])):
                    cv2.fillPoly(
                          canvas_box[idx], 
                          [np.array([corner[4],corner[5],corner[6],corner[7]])],
                          color=color_face)
                
                for i_st,i_end in self.box_skeleton:
                    if pt_out_img(corner[i_st]) and pt_out_img(corner[i_end]):
                          continue
                    cv2.line(
                        canvas_box[idx],
                        (corner[i_st, 0], corner[i_st, 1]),
                        (corner[i_end, 0], corner[i_end, 1]),
                        color =color_line,
                        thickness=thickness,
                    )
        if debug:
            for idx in idxs:
                if idx==9:
                    if data_idx:
                        cv2.imwrite('./cache/boxes_'+str(data_idx).zfill(5)+'_'+str(idx).zfill(2)+'.png',canvas_box[idx])
                    else:
                        cv2.imwrite('./cache/boxes.png',canvas_box[idx])
        return canvas_box

    def __call__(self, data_dict):
        
        new_data_dict = dict() 
        image = data_dict['image'].convert('RGB')
        input_image = self.transform(image)
        img_size = (input_image.shape[2],input_image.shape[1])
        image= image.resize(img_size)
                
        if self.gd_input_name is not None:
            gd_input = data_dict[self.gd_input_name]
            gd_input = self.transform(gd_input)
            new_data_dict['grounding_downsampler_input'] = gd_input
            if not self.is_train:
                new_data_dict[self.gd_input_name] = data_dict[self.gd_input_name].resize(img_size)

        ori_labels = data_dict['ori_labels3d']
        idxs = []
        keep = []
        for i,label in enumerate(ori_labels):
            idx = self.name_map[label]
            if idx is not None:
                idxs.append(idx)
                keep.append(i)
        cam_intrinsic = data_dict['calib']['cam_intrinsic']
        
        if len(keep) > 0:
            idxs=np.array(idxs)
            boxes3d = data_dict['boxes3d'][keep]
            dis = [xyz[0]**2+xyz[1]**2+xyz[2]**2 for xyz in boxes3d] 
            arg_id = sorted(range(len(dis)), key = lambda k: dis[k],reverse=True)
            boxes3d = boxes3d[arg_id]
            idxs = idxs[arg_id]
            corners3d = boxes3d_utils.boxes3d_to_corners3d(boxes3d, rot_axis=1)
            assert (corners3d[..., 2].mean(-1) > 0).all()
            corners3d = boxes3d_utils.crop_corners3d(corners3d,z_b=0.5,threshold=0)
            corners = boxes3d_utils.corners3d_to_corners2d(corners3d, cam_intrinsic=cam_intrinsic)
        else:
            corners = np.zeros((0, 8, 2), dtype=np.float32)

        canvas_box = self.generate_canvas_box(corners,idxs,thickness=10)   
        
        if not self.is_train:
            new_data_dict[self.bd_input_name] = canvas_box.copy()*255

        canvas_box = canvas_box.transpose(1,2,0) 
        box_downsampler_input = self.box_transform(canvas_box)
        
        scene_description = data_dict['scene_description']
        if 'rain' in scene_description.lower():
            prompt = 'rainy, realistic autonomous driving scene.'
        elif 'night' in scene_description.lower():
            prompt = 'night, realistic autonomous driving scene.'
        else:
            prompt = 'realistic autonomous driving scene.'
        
        new_data_dict['box_downsampler_input'] = box_downsampler_input
        prompt_embeds = self.prompt_embed_map[prompt]
        new_data_dict.update(
            {
                'frame_idx':data_dict['frame_idx'],
                'cam_type':data_dict['cam_type'],
                'video_length':data_dict['video_length'],
                
             })
        if 'multiview_start_idx' in data_dict:
            new_data_dict.update({'multiview_start_idx':data_dict['multiview_start_idx'],})
        
        if self.is_train:
            # tmp = self.prompt_transform(data_dict['prompts'], mode='before_pool')[0]
            # prompt_embeds = self.prompt_embed_map[data_dict['prompts']]
            new_data_dict.update(
                {
                    'image': input_image,
                    # 'input_image': input_image,
                    'prompt_embeds': prompt_embeds,
                }
            )
        else:
            
            new_data_dict.update(
                {
                    'image': image,
                    'input_image': input_image,
                    'prompt_embeds': prompt_embeds,
                    # 'prompt': data_dict['prompts'],
                    'height': input_image.shape[1],
                    'width': input_image.shape[2],
                }
            )
        
        return new_data_dict
