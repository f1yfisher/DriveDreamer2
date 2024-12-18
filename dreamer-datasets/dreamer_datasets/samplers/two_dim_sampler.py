import math

import numpy as np
import torch


class TwoDimSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        sample_first_dim=True,
        dim_size=1,
        orders=None,
        batch_size=None,
        shuffle=True,
        infinite=True,
        seed=6666,
    ):
        if orders is not None:
            assert len(orders) == dim_size
        self.sample_first_dim = sample_first_dim
        self.dim_size = dim_size
        self.orders = orders
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size = len(dataset)
        if batch_size is not None:
            self.total_size = int(math.ceil(self.data_size / batch_size)) * batch_size
        else:
            self.total_size = self.data_size
        assert self.data_size % dim_size == 0 and self.total_size % dim_size == 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_size

    def __iter__(self):
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            local_size = self.total_size // self.dim_size
            indices = np.zeros((0, self.dim_size), np.int64)
            while indices.shape[0] < local_size:
                indices_i = np.arange(self.data_size)
                if self.sample_first_dim:
                    indices_i = indices_i.reshape((-1, self.dim_size))
                else:
                    indices_i = indices_i.reshape((self.dim_size, -1)).transpose((1, 0))
                if self.shuffle:
                    indices_i = np.random.permutation(indices_i)
                num_data = min(len(indices_i), local_size - indices.shape[0])
                indices = np.concatenate([indices, indices_i[:num_data]], axis=0)
            if self.orders is not None:
                indices = indices[:, self.orders]
            indices = indices.reshape(-1)
            yield from indices
            if not self.infinite:
                break
