
The training code will be released soon.

Download the pretrained model weights [HERE](https://pan.baidu.com/s/1EPWcO_sCvlgqVFgNiDGk8w?pwd=dkjq). 

**1. Test DriveDreamer2 with first image condition (3D box and HDMap as conditions), and make visualizations.**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer2 \
        --config_name drivedreamer2_img_cond \
        --runners drivedreamer2.DriveDreamer2_Tester
```

**2. Test DriveDreamer2 with FRONT view video condition (3D box and HDMap as conditions), and make visualizations.**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer2 \
        --config_name drivedreamer2_video_cond \
        --runners drivedreamer2.DriveDreamer2_Tester
```

**3. Test DriveDreamer2 without image condition (3D box and HDMap as conditions), and make visualizations.**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer2 \
        --config_name drivedreamer2_wo_img \
        --runners drivedreamer2.DriveDreamer2_Tester
```

## Basic information of config file

<div align="center">
  
| Name |  Info |
| :----: | :----: |
| exp_dir         | Path to save logs and checkpoints |
| train_data      | The converted train dataset path (e.g., .../cam_all_train/v0.0.2) |
| test_data       | The converted test dataset path (e.g., .../cam_all_val/v0.0.2) |
| hz_factor       | The video fps = 12 / hz_factor, 12 is the fps of raw nusc camera data |
| weight_path     | Specify your weight path during testing. None is the last ckpt you trained|
| embed_map_path  | The preprocessed prompt embedding file|
</div>
