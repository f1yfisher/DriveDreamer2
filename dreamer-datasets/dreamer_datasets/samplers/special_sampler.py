import math

import numpy as np
import torch

from ..datasets import ConcatDataset


class SpecialDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=None, shuffle=True, infinite=True, seed=6666):
        assert isinstance(dataset, ConcatDataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size_list = []
        self.total_size_list = []
        for dataset in dataset.datasets:
            data_size = len(dataset)
            if batch_size is not None:
                total_size = int(math.ceil(data_size / batch_size)) * batch_size
            else:
                total_size = data_size
            self.data_size_list.append(data_size)
            self.total_size_list.append(total_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return sum(self.total_size_list)

    def __iter__(self):
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            start_idx = 0
            indices_list = []
            for i in range(len(self.data_size_list)):
                indices = np.zeros((0,), dtype=np.int64)
                data_size = self.data_size_list[i]
                total_size = self.total_size_list[i]
                while len(indices) < total_size:
                    indices_i = np.arange(data_size) + start_idx
                    if self.shuffle:
                        indices_i = np.random.permutation(indices_i)
                    num_data = min(len(indices_i), total_size - len(indices))
                    indices = np.hstack((indices, indices_i[:num_data]))
                indices_list.append(indices.reshape((-1, self.batch_size)))
                start_idx += data_size
            indices = np.concatenate(indices_list, axis=0)
            if self.shuffle:
                indices = np.random.permutation(indices)
            indices = indices.reshape(-1)
            yield from indices
            if not self.infinite:
                break
