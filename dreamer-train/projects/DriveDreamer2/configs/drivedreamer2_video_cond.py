import os
# ============= PATH ===================
proj_name = os.path.basename(__file__)[:-3]
exp_dir = '/mnt/data-2/users/zhaoguosheng/1-code/16-drivedreamer2_release/exp'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, proj_name)
train_data = '/mnt/pfs/datasets/giga_datasets/public_data/nuscenes/v1.0-trainval/cam_all_train/v0.0.2'
test_data = '/mnt/pfs/datasets/giga_datasets/public_data/nuscenes/v1.0-trainval/cam_all_val/v0.0.2'
embed_map_path = '/mnt/pfs/datasets/giga_datasets/public_data/nuscenes/clip_text_transform_after_pool_panoramic.pkl'
weight_path = '/mnt/data-2/users/zhaoguosheng/1-code/16-drivedreamer2_release/pretrained_models/drivedreamer2_video_cond/pytorch_gligen_weights.bin' # PATH TO YOUR MODEL 
# set None to load the latest checkpoint in your project_dir

save_path = '/mnt/data-2/users/zhaoguosheng/1-code/16-drivedreamer2_release/output/v1.0-trainval/drivedreamer2_video_cond' # PATH FOR SAVING GENERATED VIDEO

# ============= Data Parameters =================
resolution=(256, 448)
frame_num = 8
hz_factor = int(12/4)
view = 'CAM_FRONT' #'CAM_FRONT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'ALL'
cam_num = 6
is_multiview = cam_num>1
img_mask_type = None 
img_mask_num = 0
# ============= Model Parameters =================
gd_add_channel = 8
box_add_channel = 20
num_inf_steps = 30
sample_scheduler = 'EulerDiscreteScheduler'
mode = 'video_cond' # ['img_cond','video_cond','wo_img']
# ============= Train Parameters =================
gpu_ids = [0]
max_epochs = 10
gradient_accumulation_steps = 1
resume = False

config = dict(
    resume=resume,
    project_dir=project_dir,
    launch=dict(
        # gpu_ids=[0,1,2,3,4,5,6],
        gpu_ids=gpu_ids,
    ),
    dataloaders=dict(
        train=dict(
            type='Video',
            data_or_config=train_data,
            frame_num=frame_num,
            cam_num=cam_num,
            num_workers=2,
            transform=dict(
                type='DriveDreamer2_Transform',
                embed_map_path = embed_map_path,
                text_embed_dim = 1024,
                resolution=resolution,
                gd_input_name='image_hdmap',
                bd_input_name='image_box',
                box_normal = True,
                is_train=True,
            ),
            shuffle=True,
            hz_factor=hz_factor,
            is_multiview=is_multiview,
            view=view,
            img_mask_type=img_mask_type,
            img_mask_num=img_mask_num,
            resample=False,
        ),
        test=dict(
            type='Video',
            data_or_config=test_data,
            frame_num=frame_num,
            cam_num=cam_num,
            num_workers=0,
            transform=dict(
                type='DriveDreamer2_Transform',
                embed_map_path = embed_map_path,
                text_embed_dim = 1024,
                resolution=resolution,
                gd_input_name='image_hdmap',
                bd_input_name='image_box',
                box_normal = True,
                is_train=False,
            ),
            shuffle=True,
            hz_factor=hz_factor,
            is_multiview=is_multiview,
            view=view,
            img_mask_type=img_mask_type,
            img_mask_num=img_mask_num,
            fps = 4
        ),
    ),
    models=dict(
        pretrained='/mnt/pfs/models/huggingface/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1',
        text_encoder_pretrained='cerspense/zeroscope_v2_576w',
        weight_path=weight_path,
        mode = mode,
        enable_gradient_checkpointing=False,
        local_files_only = True,
        train_temp = True,
        pipeline_name='DriveDreamer2Pipeline',
        add_in_channels=gd_add_channel+box_add_channel,
        grounding_downsampler=dict(
            type='GroundingDownSampler',
            out_dim=gd_add_channel,
        ),
        box_downsampler=dict(
            type='GroundingDownSampler',
            in_dim=19,
            mid_dim=32,
            out_dim=box_add_channel,
        ),
        unet=dict(
            type='UNetSpatioTemporalConditionModel',
            block_out_channels=(320, 640, 1280, 1280),
            addition_time_embed_dim=256,
            num_attention_heads=(5,10,20,20),
            projection_class_embeddings_input_dim=768,
            transformer_layers_per_block=1,
            layers_per_block=2,
            sample_size=96,
            in_channels=9,
            out_channels=4,
            down_block_types=("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal"),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            cross_attention_dim=1024,
        ),
        sample_scheduler=sample_scheduler,
        num_inf_steps=num_inf_steps,
    ),
    optimizers=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=0.0,
    ),
    schedulers=dict(
        name='constant',
        num_warmup_steps=100,
    ),
    train=dict(
        max_epochs=max_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16',
        checkpoint_interval=1,
        # checkpoint_total_limit=2,
        log_with='tensorboard',
        log_interval=100,
        frame_num=frame_num,
        # max_grad_norm=1.0,
    ),
    test=dict(
        mixed_precision='fp16',
        save_dir=save_path,
        frame_num=frame_num,
        min_guidance_scale=2,
        max_guidance_scale=5,
    ),
)
