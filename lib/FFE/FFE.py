import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as Fu
import numpy as np
# import matplotlib.pyplot as plt
# plt.imshow(A[0, 0].cpu().numpy())
# plt.show()
# plt.savefig('zzz.png')

class FFE(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.T = 7

        self.ReLU = nn.ReLU()
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.a = nn.Parameter(torch.tensor([[[0., 0.]]], dtype=torch.float)).cuda()
        self.mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.float).cuda()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(512, 2 * self.T)

        self._init_weights()

    def _init_weights(self):

        init.normal_(self.linear1.weight, std=0.01)
        init.constant_(self.linear1.bias, 0)

    def forward(self, F):
        # print(self.T)
        b, c, w, h = F.shape
        F1 = self.GAP(F).view(b, -1).contiguous()
        F1 = self.sigmoid(self.linear1(F1)).view(self.T, 2).unsqueeze(-1)
        F1 = F1.expand((self.T, 2, 2)) * self.mask
        if self.T > 1:
            a = self.a.expand((self.T, 1, 2)).permute(0, 2, 1)
            F_E = F.repeat((self.T, 1, 1, 1))
        else:
            a = self.a.permute(0, 2, 1)
            F_E = F

        theta = torch.cat((F1, a), dim=2)

        grid = Fu.affine_grid(theta, F_E.size())
        trans_x = Fu.grid_sample(F_E, grid)
        return trans_x, a, F1



