import os


def get_root_dir():
    return os.path.abspath(__file__).split('giga_models')[0][:-1]


def get_model_dir():
    return os.environ.get('GIGA_MODELS_DIR', './models/')
