import os
import socket
import subprocess

from accelerate import DistributedType
from accelerate.commands.config.config_args import ClusterConfig
from accelerate.utils import ComputeEnvironment

import torch
from ..configs import load_config
from ..utils import wait_for_gpu_memory
from .run_task import run_tasks

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


class Launcher:
    def __init__(
        self,
        gpu_ids,
        machine_rank=0,
        num_machines=1,
        distributed_type=None,
        main_process_ip='127.0.0.1',
        main_process_port=None,
        num_cpu_threads_per_process=2,
        save_config_path='./cluster_config.json',
        env=None,
        executable=None,
        **kwargs,
    ):
        assert len(gpu_ids) > 0
        if distributed_type is None:
            if num_machines == 1:
                if len(gpu_ids) > 1:
                    distributed_type = DistributedType.MULTI_GPU
                else:
                    distributed_type = DistributedType.NO
        assert distributed_type is not None
        num_processes = len(gpu_ids) * num_machines
        gpu_ids = ','.join([str(i) for i in gpu_ids])
        if main_process_port is None:
            main_process_port = _find_free_port()
        if env is None:
            env = os.environ
        if executable is None:
            process = subprocess.run(['which', 'accelerate'], env=env, capture_output=True, text=True)
            if process.returncode != 0:
                raise ValueError(process.stderr)
            executable = process.stdout.strip()
        cluster_config = ClusterConfig(
            compute_environment=ComputeEnvironment.LOCAL_MACHINE,
            distributed_type=DistributedType(distributed_type),
            mixed_precision='no',
            use_cpu=False,
            debug=False,
            num_processes=num_processes,
            machine_rank=machine_rank,
            num_machines=num_machines,
            gpu_ids=gpu_ids,
            main_process_ip=main_process_ip,
            main_process_port=main_process_port,
            **kwargs,
        )
        cluster_config.to_json_file(save_config_path)
        self.cluster_config = cluster_config
        self.config_file = save_config_path
        self.num_cpu_threads_per_process = num_cpu_threads_per_process
        self.env = env
        self.executable = executable

    def launch(self, script):
        command = '{} launch'.format(self.executable)
        command += ' --config_file {}'.format(self.config_file)
        command += ' --num_cpu_threads_per_process {}'.format(self.num_cpu_threads_per_process)
        command += ' {}'.format(script)
        command = command.split(' ')
        process = subprocess.run(command, env=self.env)
        if process.returncode != 0:
            raise ValueError(process.stderr)
        
def launch_from_config(config_path, runners, gpu_memory=None, seconds=10):
    config = load_config(config_path)
    gpu_ids = config.launch.gpu_ids
    num_machines = config.launch.get('num_machines', 1)
    if gpu_memory is not None:
        local_gpu_ids = gpu_ids[0] if isinstance(gpu_ids[0], list) else gpu_ids
        wait_for_gpu_memory(local_gpu_ids, gpu_memory, unit='MB', seconds=seconds)
    if num_machines == 1 and len(gpu_ids) == 1:
        torch.cuda.set_device(gpu_ids[0])
        run_tasks(config, runners)
    else:
        launcher = Launcher(**config.launch)
        file_path = os.path.join(os.path.abspath(__file__).split('launch')[0], 'run_task.py')
        launcher.launch('{} --config {} --runners {}'.format(file_path, config_path, runners))
