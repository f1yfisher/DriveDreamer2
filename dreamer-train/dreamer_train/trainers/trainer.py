import datetime
import math
import os
import shutil
import time
from functools import reduce

import torch
from accelerate import Accelerator, DistributedType, skip_first_batches
from accelerate.utils import ProjectConfiguration, set_seed

from .. import utils
from ..configs import load_config
from ..registry import build_optimizer


class Trainer:
    def __init__(
        self,
        project_dir,
        max_epochs=0,
        max_steps=0,
        gradient_accumulation_steps=1,
        mixed_precision=None,
        checkpoint_interval=1,
        checkpoint_total_limit=-1,
        checkpoint_keeps=None,
        log_with=None,
        log_interval=100,
        activation_checkpointing=False,
        seed=6666,
        **kwargs,
    ):
        assert seed > 0
        set_seed(seed)
        if project_dir.endswith('/'):
            project_dir = project_dir[:-1]
        project_name = os.path.basename(project_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=project_dir,
            logging_dir=os.path.join(project_dir, 'logs'),
        )
        self.accelerator = Accelerator(
            split_batches=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
        )
        self.accelerator.init_trackers(project_name)
        if self.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            log_name = 'train_{}.log'.format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_total_limit = checkpoint_total_limit
        self.checkpoint_keeps = checkpoint_keeps
        self.log_interval = log_interval
        self.activation_checkpointing = activation_checkpointing
        self.seed = seed
        self.kwargs = kwargs

        self._dataloaders = []
        self._models = []
        self._optimizers = []
        self._schedulers = []

        if max_epochs > 0:
            assert max_steps == 0
            by_epoch = True
        else:
            assert max_epochs > 0
            by_epoch = False
        self._by_epoch = by_epoch
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._cur_step = 0
        self._skip_batches = 0

        self._start_tic = None
        self._epoch_tic = None
        self._step_tic = None
        self._outputs = dict()

        self.accelerator.register_for_checkpointing(self)

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
    def gradient_accumulation_steps(self):
        return self.accelerator.gradient_accumulation_steps

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

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def optimizer(self):
        return self._optimizers[0]

    @property
    def schedulers(self):
        return self._schedulers

    @property
    def scheduler(self):
        return self._schedulers[0]

    @property
    def data_size(self):
        return len(self.dataloader.dataset)

    @property
    def batch_size(self):
        if self.dataloader.batch_sampler is not None:
            batch_sampler = self.dataloader.batch_sampler
        else:
            batch_sampler = self.dataloader.sampler
        while True:
            if hasattr(batch_sampler, 'batch_sampler'):
                batch_sampler = batch_sampler.batch_sampler
            else:
                break
        batch_size = batch_sampler.batch_size
        return batch_size * self.num_processes * self.gradient_accumulation_steps

    @property
    def epoch_size(self):
        return int(math.ceil((len(self.dataloader) + self._skip_batches) / self.gradient_accumulation_steps))

    @property
    def max_epochs(self):
        if self._max_epochs > 0:
            return self._max_epochs
        else:
            return int(math.ceil(self._max_steps / self.epoch_size))

    @property
    def max_steps(self):
        if self._max_steps > 0:
            return self._max_steps
        else:
            return self._max_epochs * self.epoch_size

    @property
    def cur_epoch(self):
        return int(math.ceil(self.cur_step / self.epoch_size))

    @property
    def cur_step(self):
        return self._cur_step

    def print(self, msg, *args, **kwargs):
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    def state_dict(self):
        return {'step': self._cur_step}

    def load_state_dict(self, state_dict):
        self._cur_step = state_dict['step']

    @classmethod
    def load(cls, config_or_path):
        config = load_config(config_or_path)
        trainer = cls(project_dir=config.project_dir, **config.train)
        trainer.prepare(
            dataloaders=config.dataloaders.train,
            models=config.models.train if hasattr(config.models, 'train') else config.models,
            optimizers=config.optimizers,
            schedulers=config.schedulers,
        )
        return trainer

    def save_config(self, config):
        if not self.is_main_process:
            return
        config = load_config(config)
        config_path = os.path.join(self.project_dir, 'config.json')
        config.save(config_path)

    def get_checkpoint(self, checkpoint=None, model_name=None):
        if checkpoint is None:
            checkpoints = os.listdir(self.model_dir)
            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
            if len(checkpoints) > 0:
                checkpoint = os.path.join(self.model_dir, checkpoints[-1])
            else:
                return None
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]
        for i in range(len(checkpoint)):
            if checkpoint[i].startswith('checkpoint'):
                checkpoint[i] = os.path.join(self.model_dir, checkpoint[i])
            if model_name is not None and os.path.isdir(checkpoint[i]):
                checkpoint[i] = os.path.join(checkpoint[i], model_name)
            assert os.path.exists(checkpoint[i])
        return checkpoint if len(checkpoint) > 1 else checkpoint[0]

    def remove_checkpoint(self, total_limit=None):
        if not self.is_main_process:
            return
        total_limit = total_limit or self.checkpoint_total_limit
        checkpoints = os.listdir(self.model_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
        if self.checkpoint_keeps is not None:
            new_checkpoints = []
            for checkpoint in checkpoints:
                if self._by_epoch:
                    checkpoint_id = int(checkpoint.split('_')[-3])
                else:
                    checkpoint_id = int(checkpoint.split('_')[-1])
                if checkpoint_id not in self.checkpoint_keeps:
                    new_checkpoints.append(checkpoint)
            checkpoints = new_checkpoints
        if len(checkpoints) >= total_limit > 0:
            num_to_remove = len(checkpoints) - self.checkpoint_total_limit + 1
            for checkpoint in checkpoints[:num_to_remove]:
                checkpoint = os.path.join(self.model_dir, checkpoint)
                self.logger.info('Remove checkpoint {}'.format(checkpoint))
                shutil.rmtree(checkpoint)

    def resume(self, checkpoint=None):
        checkpoint = self.get_checkpoint(checkpoint)
        if checkpoint is None:
            return
        self.accelerator.load_state(checkpoint)
        if self.dataloader.batch_sampler is not None:
            sampler = self.dataloader.batch_sampler
        else:
            sampler = self.dataloader.sampler
        while True:
            if hasattr(sampler, 'batch_sampler'):
                sampler = sampler.batch_sampler
            elif hasattr(sampler, 'sampler'):
                sampler = sampler.sampler
            else:
                break
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(int(math.floor(self.cur_step / self.epoch_size)))
            skip_batches = (self.cur_step % self.epoch_size) * self.gradient_accumulation_steps
        else:
            skip_batches = self.cur_step * self.gradient_accumulation_steps
        if skip_batches > 0:
            for i in range(len(self._dataloaders)):
                self._dataloaders[i] = skip_first_batches(self._dataloaders[i], skip_batches)
        self._skip_batches = skip_batches

    def get_dataloaders(self, *args, **kwargs):
        raise NotImplementedError

    def get_models(self, *args, **kwargs):
        raise NotImplementedError

    def get_optimizers(self, optimizers):
        optimizers = utils.as_list(optimizers)
        for i in range(len(optimizers)):
            if isinstance(optimizers[i], dict):
                if len(optimizers) == 1 and len(self.models) > 1:
                    params = []
                    for model in self.models:
                        params += list(model.parameters())
                elif len(optimizers) == len(self.models):
                    params = self.models[i].parameters()
                else:
                    assert False
                optimizers[i] = build_optimizer(optimizers[i], params=params)
        return optimizers

    def get_schedulers(self, *args, **kwargs):
        raise NotImplementedError

    def prepare(self, dataloaders, models, optimizers, schedulers):
        self._dataloaders = utils.as_list(self.get_dataloaders(dataloaders))
        self._models = utils.as_list(self.get_models(models))
        if self.distributed_type == DistributedType.FSDP:
            self._models = utils.as_list(self.accelerator.prepare(*self._models))
        self._optimizers = utils.as_list(self.get_optimizers(optimizers))
        self._schedulers = utils.as_list(self.get_schedulers(schedulers))
        if self.distributed_type == DistributedType.FSDP:
            objects = [self._dataloaders, self._optimizers, self._schedulers]
        else:
            objects = [self._dataloaders, self._models, self._optimizers, self._schedulers]
        inputs = reduce(lambda x, y: x + y, objects)
        outputs = utils.as_list(self.accelerator.prepare(*inputs))
        start_idx = 0
        for obj in objects:
            end_idx = start_idx + len(obj)
            obj[:] = outputs[start_idx:end_idx]
            start_idx = end_idx

    def train(self):
        self.print_before_train()
        dataloader_iter = iter(self.dataloader)
        for self._cur_step in range(self._cur_step, self.max_steps):
            self._cur_step += 1
            for _ in range(self.gradient_accumulation_steps):
                batch_dict = next(dataloader_iter)
                with self.accelerator.accumulate(*self.models):
                    losses = self.forward_step(batch_dict)
                    loss = self.parse_losses(losses)
                    self.backward_step(loss)
            self.print_step()
            self.save_checkpoint_step()
        self.print_after_train()
        self.accelerator.end_training()

    def forward_step(self, batch_dict):
        return self.model(batch_dict)

    def backward_step(self, loss):
        self.accelerator.backward(loss)
        max_grad_norm = self.kwargs.get('max_grad_norm', None)
        grad_norm_type = self.kwargs.get('grad_norm_type', 2)
        if self.accelerator.sync_gradients and max_grad_norm is not None:
            params = []
            for model in self.models:
                params += list(model.parameters())
            self.accelerator.clip_grad_norm_(params, max_grad_norm, grad_norm_type)
        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def save_checkpoint_step(self):
        if self._by_epoch:
            checkpoint_interval = int(self.checkpoint_interval * self.epoch_size)
        else:
            checkpoint_interval = int(self.checkpoint_interval)
        if self.cur_step % checkpoint_interval == 0 or self.cur_step == self.max_steps:
            output_name = 'checkpoint_epoch_{}_step_{}'.format(self.cur_epoch, self.cur_step)
            output_dir = os.path.join(self.model_dir, output_name)
            if self.is_main_process:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                self.remove_checkpoint()
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(output_dir)

    def print_before_train(self):
        msg = 'num_processes: {}'.format(self.num_processes)
        msg += ', process_index: {}'.format(self.process_index)
        msg += ', data_size: {}'.format(self.data_size)
        msg += ', batch_size: {}'.format(self.batch_size)
        msg += ', epoch_size: {}'.format(self.epoch_size)
        self.logger.info(msg)
        self._epoch_tic = self._step_tic = self._start_tic = time.time()

    def print_step(self):
        if not self.is_main_process:
            return
        if self.cur_step % self.log_interval == 0:
            outputs = dict()
            for key, val in self._outputs.items():
                val = val['sum'] / val['num'] if val['num'] > 0 else float('nan')
                outputs[key] = val
            self._outputs.clear()
            self.accelerator.log(outputs, self.cur_step)
            time_cost = time.time() - self._step_tic
            self._step_tic = time.time()
            speed = self.log_interval * self.batch_size / time_cost
            eta_sec = max(0, time_cost / self.log_interval * (self.max_steps - self.cur_step))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            lr = self.scheduler.get_last_lr()[0]
            if self._by_epoch:
                inner_step = (self.cur_step - 1) % self.epoch_size + 1
                msg = 'Epoch[%d/%d][%d/%d]' % (self.cur_epoch, self.max_epochs, inner_step, self.epoch_size)
            else:
                msg = 'Step[%d/%d]' % (self.cur_step, self.max_steps)
            msg += ' eta: %s, time: %.3f, speed: %.3f, lr: %.3e' % (eta_str, time_cost, speed, lr)
            if self.accelerator.scaler is not None and self.accelerator.scaler.is_enabled():
                msg += ', grad_scale: %.3f' % self.accelerator.scaler.get_scale()
            for key, val in outputs.items():
                msg += ', %s: %.3f' % (key, val)
            self.logger.info(msg)
        if self._by_epoch and self.cur_step % self.epoch_size == 0:
            time_cost = time.time() - self._epoch_tic
            time_cost = str(datetime.timedelta(seconds=int(time_cost)))
            self._epoch_tic = time.time()
            self.logger.info('Total_time: %s' % time_cost)

    def print_after_train(self):
        if not self.is_main_process:
            return
        time_cost = time.time() - self._start_tic
        time_cost = str(datetime.timedelta(seconds=int(time_cost)))
        self.logger.info('Total_time: %s' % time_cost)

    def parse_losses(self, losses):
        if isinstance(losses, dict):
            assert 'total_loss' not in losses
            for key, val in losses.items():
                losses[key] = val.mean()
            loss = sum(losses.values())
            outputs = {'total_loss': loss}
            outputs.update(losses)
        elif isinstance(losses, torch.Tensor):
            loss = losses.mean()
            outputs = {'total_loss': loss}
        else:
            assert False
        for key, val in outputs.items():
            if key not in self._outputs:
                self._outputs[key] = {'sum': 0.0, 'num': 0}
            self._outputs[key]['sum'] += val.item()
            self._outputs[key]['num'] += 1
        return loss
