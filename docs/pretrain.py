import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import anndata as ad
from tqdm import tqdm
from typing import Literal

from .model import SCNetAE
from ._utils import seed_everything


def pretrain(adata: ad.AnnData,
             n_epochs: int = 50,
             batch_size: int = 128,
             learning_rate: float = 1e-5,
             GPU: bool = True,
             random_state: int = None,
             norm_type: Literal['Batch', 'Instance'] = 'Batch',
             net_type: Literal['MLP', 'Conv1d'] = 'MLP',
             ):
    if GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        seed_everything(random_state)
    
    # Initialize dataloader for train data
    train_data = torch.as_tensor(torch.from_numpy(adata.X), dtype=torch.float32)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    net = SCNetAE(adata.n_vars, norm_type=norm_type, net_type=net_type).to(device)
    opt_G = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    G_scaler = torch.cuda.amp.GradScaler()
    L1 = nn.L1Loss().to(device)

    net.train()
    with tqdm(total=n_epochs) as t:
        for _ in range(n_epochs):
            t.set_description(f'Pretrain {net_type}')

            for _, data in enumerate(train_loader):
                data = data.to(device)

                re_data = net(data)
                Loss = L1(data, re_data)
                opt_G.zero_grad()
                G_scaler.scale(Loss).backward()
                G_scaler.step(opt_G)
                G_scaler.update()

            t.set_postfix(Loss = Loss.item())
            t.update(1)
    
    return net.state_dict()