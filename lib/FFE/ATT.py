import torch
import torch.nn as nn
import torch.nn.init as init


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3 or 7'
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = 3

        self.att3 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.att3.weight, std=0.01)
        init.constant_(self.att3.bias, 0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.att3(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)
