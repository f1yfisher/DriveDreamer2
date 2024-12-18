import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config

class GroundingDownSampler(nn.Module,ConfigMixin):
    config_name='config.json'
    @register_to_config
    def __init__(self, in_dim=3, mid_dim=4, out_dim=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_dim, mid_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_dim, out_dim, 4, 2, 1),
        )
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        return self.layers(x)
