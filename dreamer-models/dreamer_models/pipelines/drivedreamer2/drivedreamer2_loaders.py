import torch
from ...models.drivedreamer2 import grounding_downsampler as gd_models
from ...models.drivedreamer2 import unet_spatio_temporal_condition



class DriveDreamer2_LoaderMixin:
    def load_weights(self, pretrained_model_name_path):
        # TODO  此处只load了训练参数的部分,需要check我们训练的时候的训练参数有哪些
        device = self.unet.device
        dtype = self.unet.dtype
        state_dict = torch.load(pretrained_model_name_path, map_location='cpu')
        meta = state_dict.pop('meta')
        grounding_downsampler = meta.get('grounding_downsampler', None)
        img_cond_downsampler = meta.get('img_cond_downsampler', None)
        box_downsampler = meta.get('box_downsampler',None)
       
        unet2d_state_dict = self.unet.state_dict()
        unet_ = meta.get('unet', None)

        self.unet = getattr(unet_spatio_temporal_condition, unet_.pop('_class_name'))(**unet_)

        unet3d_state_dict = self.unet.state_dict()
        
        # load ckpt from 2d stable diffusion unet model
        missing_keys_dict = {}
        for name, param in unet3d_state_dict.items():
            if name in unet2d_state_dict.keys():
                if unet2d_state_dict[name].shape != unet3d_state_dict[name].shape:
                    try:
                        unet2d_state_dict[name] = unet2d_state_dict[name].reshape(unet3d_state_dict[name].shape)
                    except:
                        missing_keys_dict[name] = True
                        continue
                unet3d_state_dict[name] = unet2d_state_dict[name]
            else:
                missing_keys_dict[name] = True

        # load ckpt from 3d unet model
        gd_state_dict = dict()
        img_cond_state_dict = dict()
        bd_state_dict = dict()
        for name, param in state_dict.items():
            if name.startswith('unet.'):
                name = name[len('unet.') :]
                if param.shape != unet3d_state_dict[name].shape:
                    param = param.reshape(unet3d_state_dict[name].shape)
                unet3d_state_dict[name] = param
                try:
                    missing_keys_dict.pop(name)
                except KeyError:
                    continue
            elif name.startswith('grounding_downsampler.'):
                name = name[len('grounding_downsampler.') :]
                gd_state_dict[name] = param
            elif name.startswith('box_downsampler.'):
                name = name[len('box_downsampler.') :]
                bd_state_dict[name]=param
            elif name.startswith('img_cond_downsampler.'):
                name = name[len('img_cond_downsampler.') :]
                img_cond_state_dict[name] = param
            else:
                assert False
        assert len(missing_keys_dict) == 0, f"missing keys: {missing_keys_dict.keys()}"
        self.unet.load_state_dict(unet3d_state_dict, strict=True)
        self.unet.to(device, dtype=dtype)
        if grounding_downsampler is not None:
            class_name = grounding_downsampler.pop('_class_name')
            grounding_downsampler = getattr(gd_models, class_name)(**grounding_downsampler)
            grounding_downsampler.load_state_dict(gd_state_dict)
            grounding_downsampler.to(device, dtype=dtype)
        else:
            assert len(gd_state_dict) == 0
        self.grounding_downsampler = grounding_downsampler
        if box_downsampler is not None:
            class_name = box_downsampler.pop('_class_name')
            box_downsampler = getattr(gd_models, class_name)(**box_downsampler)
            box_downsampler.load_state_dict(bd_state_dict)
            box_downsampler.to(device, dtype=dtype)
        else:
            assert len(bd_state_dict) == 0
        self.box_downsampler = box_downsampler
        if img_cond_downsampler is not None:
            class_name = img_cond_downsampler.pop('_class_name')
            img_cond_downsampler = getattr(gd_models, class_name)(**img_cond_downsampler)
            img_cond_downsampler.load_state_dict(img_cond_state_dict)
            img_cond_downsampler.to(device, dtype=dtype)
            self.img_cond_downsampler = img_cond_downsampler
        else:
            assert len(img_cond_state_dict) == 0

    

