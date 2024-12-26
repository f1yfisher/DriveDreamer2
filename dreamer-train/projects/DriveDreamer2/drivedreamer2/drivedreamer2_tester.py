import copy
import os
import math
import shutil
import time
import imageio
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import DDIMScheduler, UniPCMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,AutoencoderKLTemporalDecoder

from dreamer_datasets import DefaultCollator, load_dataset, DefaultSampler,CLIPTextTransform
from dreamer_models import DriveDreamer2Pipeline
from dreamer_train import Tester
from . import drivedreamer2_transforms
from .drivedreamer2_utils import GLIGEN_WEIGHT_NAME, VideoSampler, VideoCollator, draw_mv_video,draw_mv_video_v2

CAM_NAMES = ['CAM_FRONT',
'CAM_FRONT_LEFT',
'CAM_FRONT_RIGHT',
'CAM_BACK',
'CAM_BACK_LEFT',
'CAM_BACK_RIGHT']
from torchvision import transforms

class DriveDreamer2_Tester(Tester):
    def get_dataloaders(self, data_config):
        self.data_config = data_config
        dataset = load_dataset(data_config.data_or_config)
        transform = getattr(drivedreamer2_transforms, data_config.transform.pop('type'))(**data_config.transform)
        dataset.set_transform(transform)

        self.fps=data_config.fps
        self.cam_num = data_config.cam_num
        self.frame_num = data_config.frame_num
        batch_size_per_gpu = self.frame_num * self.cam_num
        cam_names = data_config.get('cam_names',None)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=VideoSampler(
                    dataset, 
                    batch_size=batch_size_per_gpu, 
                    frame_num=self.frame_num, 
                    cam_num=self.cam_num,
                    video_split_rate=data_config.get('video_split_rate',1),
                    hz_factor=data_config.hz_factor,
                    mv_video=data_config.is_multiview, 
                    view=data_config.view,
                    shuffle=data_config.shuffle,
                    logger=self.logger),
            collate_fn=VideoCollator(
                frame_num=self.frame_num,
                img_mask_type=data_config.img_mask_type,
                img_mask_num=data_config.img_mask_num) 
                if 'Video' in data_config.type else DefaultCollator(),
                batch_size=batch_size_per_gpu,
                num_workers=data_config.num_workers,)
       
       
        return dataloader
    
    def get_models(self, model_config):
        local_files_only = model_config.get('local_files_only', True)
        pipeline_name = model_config.pipeline_name
        text_encoder_pretrained = model_config.get('text_encoder_pretrained',None)
        variant = 'fp16' if self.mixed_precision == 'fp16' else None
        if pipeline_name == 'DriveDreamer2Pipeline':
            model=DriveDreamer2Pipeline.from_pretrained(
                model_config.pretrained,
                torch_dtype=self.dtype,
                variant=variant,
                local_files_only=local_files_only,
                safety_checker=None,
            )
            if text_encoder_pretrained is None:
                assert False
            self.text_encoder = CLIPTextTransform(
                model_path=text_encoder_pretrained,
                device=self.device,
                dtype=self.dtype,
            )
            setattr(model, 'frame_num',8)
            # setattr(model, 'cam_num', self.cam_num)
            # model.load_clipTextTransformer()
        else:
            assert False
        
        self.mode = model_config.get('mode','img_cond')
        
        assert self.mode in ['img_cond','video_cond','wo_img']

        self.num_inf_steps = model_config.get('num_inf_steps', 50)
        
        weight_path = model_config.get('weight_path', None)
        if weight_path is None:
            checkpoint = self.get_checkpoint()
            weight_path = os.path.join(checkpoint, GLIGEN_WEIGHT_NAME)
        elif os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, GLIGEN_WEIGHT_NAME)
        
        assert weight_path is not None
        self.logger.info('load from {}'.format(weight_path))
        model.load_weights(weight_path)
        model.to(self.device)
        return model

    def test(self):
        if self.is_main_process:
            save_dir = self.kwargs.get('save_dir', None)
            os.makedirs(save_dir,exist_ok=True)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            idx = 0
            prompts = [
                ['realistic autonomous driving scene, panoramic videos from different perspectives.' ],
                ['rainy, realistic autonomous driving scene, panoramic videos from different perspectives.'],
                ['night, realistic autonomous driving scene, panoramic videos from different perspectives.'],
            ]
            for batch_dict in self.dataloader:
                grounding_downsampler_input = batch_dict.get('grounding_downsampler_input', None)
                grounding_downsampler_input = grounding_downsampler_input.reshape(self.cam_num,self.frame_num,*grounding_downsampler_input.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                box_downsampler_input = batch_dict.get('box_downsampler_input',None)
                box_downsampler_input = box_downsampler_input.reshape(self.cam_num,self.frame_num,*box_downsampler_input.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                img_cond = batch_dict.get('input_image',None)
                img_cond = img_cond.reshape(self.cam_num,self.frame_num,*img_cond.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                input_dict = {
                    'grounding_downsampler_input': grounding_downsampler_input,
                    'box_downsampler_input': box_downsampler_input}
                
                if self.mode == 'img_cond':
                    input_dict.update({
                        'img_cond':img_cond[:1],
                    })
                elif self.mode =='video_cond':
                    input_dict.update({
                        'video_cond':img_cond,
                    })
                
                prompt_embed = batch_dict.get('prompt_embeds',None)
                if prompt_embed is None:
                    videos = []
                    for this_prompt in prompts:
                        this_prompt_embed = self.text_encoder(this_prompt,mode='after_pool',to_numpy=False)[:,None]
                        images = self.model(
                            this_prompt_embed,
                            scheduled_sampling_beta=1.0,
                            input_dict=copy.deepcopy(input_dict),
                            height=batch_dict['height'][0],
                            width=batch_dict['width'][0]*6,
                            generator=generator,
                            min_guidance_scale=self.kwargs.get('min_guidance_scale', 1),
                            max_guidance_scale=self.kwargs.get('max_guidance_scale', 7.5),
                            num_inference_steps=self.num_inf_steps,
                            num_frames=self.frame_num,
                            first_frame=True,
                        )
                        
                        images=images.frames[0]
                        videos.append(images)

                    if save_dir is not None:
                        images = draw_mv_video_v2(videos, batch_dict)
                        imageio.mimsave(os.path.join(save_dir, '{:06d}.mp4'.format(idx)), images, fps=self.fps)
                        idx += 1
                else:
                    images = self.model(
                                prompt_embed[0:1].half(),
                                scheduled_sampling_beta=1.0,
                                input_dict=copy.deepcopy(input_dict),
                                height=batch_dict['height'][0],
                                width=batch_dict['width'][0]*6,
                                generator=generator,
                                min_guidance_scale=self.kwargs.get('min_guidance_scale',1),
                                max_guidance_scale=self.kwargs.get('max_guidance_scale', 7.5),
                                num_inference_steps=self.num_inf_steps,
                                num_frames=self.frame_num,
                                first_frame=True,
                            )
                    images=images.frames[0]
                    if save_dir is not None:
                        images = draw_mv_video(images, batch_dict)
                        imageio.mimsave(os.path.join(save_dir, '{:06d}.mp4'.format(idx)), images, fps=self.fps)
                    idx += 1     
        self.accelerator.wait_for_everyone()
