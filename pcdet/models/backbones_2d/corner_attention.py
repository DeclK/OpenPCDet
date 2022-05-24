import torch
import torch.nn as nn

from functools import partial

# corner classification module
class CCM(nn.Module):
    def __init__(self, input_channels, num_class) -> None:
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.shared_fc = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1),
            norm_fn(64),
            nn.ReLU()
        )

        self.corner_cls = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            norm_fn(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            norm_fn(64),
            nn.ReLU(),
            nn.Conv2d(64, num_class * 3, 1)
        )

        self.corner_reg = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            norm_fn(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            norm_fn(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )

    def build_target(self,):
        # positive & negative for regression
        pass

    def get_loss(self,):
        pass

    def forward(self, batch_dict):
        pass