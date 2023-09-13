import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from ._block import MemoryBlock, StyleBlock


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm: bool = True, act: bool = True,
                 norm_type=Literal['Batch', 'Instance']):
        super().__init__()
        if norm_type == 'Batch':
            self.linear = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            )
        elif norm_type == 'Instance':
            self.linear = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.InstanceNorm1d(out_dim) if norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            )

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv1dBlock(nn.Module):
    def __init__(self, in_chan, out_chan, Transpose: bool = False,
                 norm: bool = True, act: bool = True,
                 norm_type=Literal['Batch', 'Instance']):
        super().__init__()
        if norm_type == 'Batch':
            self.conv = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, 3, 2, 1)
            if not Transpose else
            nn.ConvTranspose1d(in_chan, out_chan, 3, 2, 1, 1),
            nn.BatchNorm1d(out_chan) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            )
        elif norm_type == 'Instance':
            self.conv = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, 3, 2, 1)
            if not Transpose else
            nn.ConvTranspose1d(in_chan, out_chan, 3, 2, 1, 1),
            nn.InstanceNorm1d(out_chan) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], norm_type='Batch'):
        super().__init__()
        self.down = nn.Sequential(
            LinearBlock(in_dim, out_dim[0], norm_type=norm_type),
            LinearBlock(out_dim[0], out_dim[1], norm_type=norm_type),
            LinearBlock(out_dim[1], out_dim[2], act=False, norm_type=norm_type),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Decoder_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], norm_type='Batch'):
        super().__init__()
        self.up = nn.Sequential(
            LinearBlock(out_dim[2], out_dim[1], norm_type=norm_type),
            LinearBlock(out_dim[1], out_dim[0], norm_type=norm_type),
            LinearBlock(out_dim[0], in_dim, norm=False, act=False, norm_type=norm_type),
        )

    def forward(self, x):
        x = self.up(x)
        return F.relu(x)


class Encoder_Conv1d(nn.Module):
    def __init__(self, in_chan: int = 1, feat: int = 32, norm_type='Batch'):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_chan, feat, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down = nn.Sequential(
            Conv1dBlock(feat, feat*2, norm_type=norm_type),
            Conv1dBlock(feat*2, feat*4, norm_type=norm_type),
            Conv1dBlock(feat*4, feat*4, norm_type=norm_type),
        )

        self.last = nn.Conv1d(feat*4, 1, 4, 1, 1)

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        return self.last(x)


class Decoder_Conv1d(nn.Module):
    def __init__(self, in_chan: int = 1, feat: int = 32, norm_type='Batch'):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose1d(1, feat*4, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up = nn.Sequential(
            Conv1dBlock(feat*4, feat*4, True, norm_type=norm_type),
            Conv1dBlock(feat*4, feat*2, True, norm_type=norm_type),
            Conv1dBlock(feat*2, feat, True, norm_type=norm_type),
        )

        self.last = nn.ConvTranspose1d(feat, in_chan, 7, 1, 3)

    def forward(self, x):
        x = self.initial(x)
        x = self.up(x)
        return F.relu(self.last(x))


class SCNetAE(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256],
                 in_chan: int = 1, feat: int =32,
                 norm_type: Literal['Batch', 'Instance'] = 'Batch', 
                 net_type: Literal['MLP', 'Conv1d'] = 'MLP'):
        super().__init__()
        if net_type == 'MLP':
            self.encoder = Encoder_MLP(in_dim, out_dim, norm_type)
            self.decoder = Decoder_MLP(in_dim, out_dim, norm_type)
        elif net_type == 'Conv1d':
            self.encoder = Encoder_Conv1d(in_chan, feat, norm_type)
            self.decoder = Decoder_Conv1d(in_chan, feat, norm_type)

    def encode(self, x):
        x = x.unsqueeze(dim=1)
        return self.encoder(x).squeeze(dim=1)

    def decode(self, z):
        z = z.unsqueeze(dim=1)
        return self.decoder(z).squeeze(dim=1)

    def forward(self, x):
        return self.decode(self.encode(x))


class Memory_G(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256],
                 in_chan: int = 1, feat: int =32,
                 net_type: Literal['MLP', 'Conv1d'] = 'MLP',
                 mem_dim=2048, thres=0.005, temperature=0.5):
        super().__init__()
        if net_type == 'Conv1d':
            z_dim = int(in_dim/8) - 1
        elif net_type == 'MLP':
            z_dim = out_dim[2]

        self.net = SCNetAE(in_dim, out_dim, in_chan, feat, 'Batch', net_type)
        self.Memory = MemoryBlock(mem_dim, z_dim, thres, temperature)

    def forward(self, x):
        real_z = self.net.encode(x)
        mem_z = self.Memory(real_z)
        fake_x = self.net.decode(mem_z)
        fake_z = self.net.encode(fake_x)
        return real_z, fake_x, fake_z


class Align_G(nn.Module):
    def __init__(self, base_cells, input_cells, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.net = SCNetAE(in_dim, out_dim, 'Batch')
        self.mapping = nn.Parameter(torch.Tensor(base_cells, input_cells))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mapping.size(1))
        self.mapping.data.uniform_(-stdv, stdv)

    def forward(self, x, base):
        z = self.net.encode(x)
        z = F.normalize(z, p=1, dim=1)
        fake_z = torch.mm(F.relu(self.mapping), z)
        z = self.net.encode(base)
        z = F.normalize(z, p=1, dim=1)
        return fake_z, z, F.relu(self.mapping)


class Batch_G(nn.Module):
    def __init__(self, data_n, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.net = SCNetAE(in_dim, out_dim, 'Instance')
        self.Style = StyleBlock(data_n, out_dim[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, label):
        z = self.net.encode(x)
        z = self.Style(z, label)
        x = self.net.decode(z)
        return x