import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from .. import utils
from ..configs import Config


class Tester:
    def __init__(
        self,
        project_dir,
        mixed_precision=None,
        seed=6666,
        **kwargs,
    ):
        assert seed > 0
        set_seed(seed)
        accelerator_project_config = ProjectConfiguration(
            project_dir=project_dir,
            logging_dir=os.path.join(project_dir, 'logs'),
        )
        self.accelerator = Accelerator(
            split_batches=False,
            mixed_precision=mixed_precision,
            project_config=accelerator_project_config,
        )
        os.makedirs(self.logging_dir, exist_ok=True)
        if self.is_main_process:
            log_name = 'test_{}.log'.format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.seed = seed
        self.kwargs = kwargs

        self._dataloaders = []
        self._models = []

    @property
    def project_dir(self):
        return self.accelerator.project_dir

    @property
    def logging_dir(self):
        return self.accelerator.logging_dir

    @property
    def model_dir(self):
        return os.path.join(self.project_dir, 'models')

    @property
    def distributed_type(self):
        return self.accelerator.distributed_type

    @property
    def num_processes(self):
        return self.accelerator.num_processes

    @property
    def process_index(self):
        return self.accelerator.process_index

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    @property
    def is_last_process(self):
        return self.accelerator.is_last_process

    @property
    def mixed_precision(self):
        return self.accelerator.mixed_precision

    @property
    def device(self):
        return self.accelerator.device

    @property
    def dtype(self):
        return torch.float16 if self.mixed_precision == 'fp16' else torch.float32

    @property
    def dataloaders(self):
        return self._dataloaders

    @property
    def dataloader(self):
        return self._dataloaders[0]

    @property
    def models(self):
        return self._models

    @property
    def model(self):
        return self._models[0]

    def print(self, msg, *args, **kwargs):
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    @classmethod
    def load(cls, config_or_path):
        if isinstance(config_or_path, str):
            if os.path.isdir(config_or_path):
                config_path = os.path.join(config_or_path, 'config.json')
            else:
                config_path = config_or_path
            config = Config.load(config_path)
        elif isinstance(config_or_path, dict):
            config = Config(config_or_path)
        else:
            assert False
        tester = cls(project_dir=config.project_dir, **config.test)
        tester.prepare(
            dataloaders=config.dataloaders.test,
            models=config.models.test if hasattr(config.models, 'test') else config.models,
        )
        return tester

    def get_checkpoint(self, checkpoint=None, model_name=None):
        if checkpoint is None:
            checkpoints = os.listdir(self.model_dir)
            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
            if len(checkpoints) > 0:
                checkpoint = os.path.join(self.model_dir, checkpoints[-1])
            else:
                return None
        if checkpoint.startswith('checkpoint'):
            checkpoint = os.path.join(self.model_dir, checkpoint)
        if model_name is not None and os.path.isdir(checkpoint):
            checkpoint = os.path.join(checkpoint, model_name)
        assert os.path.exists(checkpoint)
        return checkpoint

    def get_dataloaders(self, *args, **kwargs):
        raise NotImplementedError

    def get_models(self, *args, **kwargs):
        raise NotImplementedError

    def prepare(self, dataloaders, models):
        self._dataloaders = utils.as_list(self.get_dataloaders(dataloaders))
        self._models = utils.as_list(self.get_models(models))
        self._dataloaders = utils.as_list(self.accelerator.prepare(*self._dataloaders))

    def test(self):
        raise NotImplementedError
