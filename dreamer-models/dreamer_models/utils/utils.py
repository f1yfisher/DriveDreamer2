import os

import torch


def wrap_call(func):
    def f(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                continue

    return f


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(int(os.environ.get('GIGA_MODELS_DEFAULT_DEVICE', '0'))))
    else:
        device = torch.device('cpu')
    return device


def is_offline_mode():
    ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
    return True if os.environ.get('GIGA_MODELS_OFFLINE', '1').upper() in ENV_VARS_TRUE_VALUES else False

def load_state_dict(weight_path):
    if os.path.isdir(weight_path):
        if os.path.exists(os.path.join(weight_path, WEIGHTS_NAME)):
            return torch.load(os.path.join(weight_path, WEIGHTS_NAME), map_location='cpu')
        elif os.path.exists(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME)):
            return safetensors.torch.load_file(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME), device='cpu')
        else:
            assert False
    elif os.path.isfile(weight_path):
        if weight_path.endswith(WEIGHTS_NAME):
            return torch.load(weight_path, map_location='cpu')
        elif weight_path.endswith(SAFETENSORS_WEIGHTS_NAME):
            return safetensors.torch.load_file(weight_path, device='cpu')
        elif weight_path.endswith('safetensors'):
            return safetensors.torch.load_file(weight_path, device='cpu')
        else:
            assert False
    else:
        assert False(drive)
