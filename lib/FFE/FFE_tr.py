import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as Fu


class FFE(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.T = 4
        self.ReLU = nn.ReLU()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(512, 2 * self.T)
        self.tanh = nn.Tanh()

        self.a = nn.Parameter(torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float))

        self._init_weights()

    def _init_weights(self):

        init.normal_(self.linear1.weight, std=0.01)
        init.constant_(self.linear1.bias, 0)

    def forward(self, F):
        A2 = F
        b, c, _, _ = A2.shape
        B = self.GAP(A2).view(b, -1).contiguous()
        B = self.tanh(self.linear1(B)).view(self.T, 2).unsqueeze(-1)

        if self.T > 1:
            a = self.a.expand((self.T, 2, 2))
            A_E = A2.repeat((self.T, 1, 1, 1))
        else:
            a = self.a
            A_E = A2

        theta = torch.cat((a, B), dim=2)
        grid = Fu.affine_grid(theta, A_E.size())
        trans_x = Fu.grid_sample(A_E, grid)
        return trans_x, a, B
















