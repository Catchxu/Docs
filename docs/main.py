import os
import pandas as pd
import anndata as ad
from tqdm import tqdm
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import Memory_G, Align_G, Batch_G
from .model import Discriminator
from ._utils import seed_everything, calculate_gradient_penalty


class DetectOutlier:
    def __init__(self, n_epochs: int = 10, batch_size: int = 128,
                 learning_rate: float = 2e-4, mem_dim: int = 2048,
                 shrink_thres: float = 0.01, temperature: float = 1,
                 n_critic: int = 1, GPU: bool = True,
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None,
                 net_type: Literal['MLP', 'Conv1d'] = 'MLP',):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train ODBC-GAN.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.tem = temperature
        self.n_critic = n_critic
        self.net_type = net_type

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_enc': 1, 'w_gp': 10}
        else:
            self.weight = weight
    
    def fit(self, adata: ad.AnnData, weight_dir: Optional[str] = None):
        tqdm.write('Begin to fine-tune the model on normal cells...')

        self.genes = adata.var_names
        train_data = torch.Tensor(adata.X)
        self.loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

        self.D = Discriminator(in_dim=adata.n_vars).to(self.device)
        self.G = Memory_G(in_dim=adata.n_vars, net_type=self.net_type, thres=self.shrink_thres,
                          mem_dim=self.mem_dim, temperature=self.tem).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()

        self.L1 = nn.L1Loss().to(self.device)
        self.L2 = nn.MSELoss().to(self.device)

        self.prepare(weight_dir)

        self.D.train()
        self.G.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, data in enumerate(self.loader):
                    data = data.to(self.device)

                    # Update discriminator for n_critic times
                    for _ in range(self.n_critic):
                        self.update_D(data)

                    # Update generator for one time
                    self.update_G(data)

                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        tqdm.write('Fine-tuning has been finished.')

    @torch.no_grad()
    def predict(self, test: ad.AnnData):
        if (test.var_names != self.genes).any():
            raise RuntimeError('Test data and train data have different genes.')

        if (self.G is None or self.D is None):
            raise RuntimeError('Please fine-tune the model first.')
        
        test_data = torch.Tensor(test.X)
        self.loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.G.eval()
        tqdm.write('Detect outlier cells on test dataset...')
        # Store real_z and res only for detecting outlier subtype
        real_z, res, score = [], [], []
        for _, data in enumerate(self.loader):
            data = data.to(self.device)
            z, fake_data, fake_z = self.G(data)

            real_z.append(z.cpu().detach())
            res.append(data.cpu() - fake_data.cpu().detach())
            score.append(1 - F.cosine_similarity(z, fake_z).reshape(-1, 1).cpu().detach())
        
        self.real_z = torch.cat(real_z, dim=0)
        self.res = torch.cat(res, dim=0)

        score = torch.cat(score, dim=0).numpy()
        # Normalize outlier scores
        score = (score - score.min())/(score.max() - score.min())
        result = pd.DataFrame({'cell_idx': test.obs_names,
                               'score': score.reshape(-1)})

        tqdm.write('Outlier cells have been detected.')
        return result

    def prepare(self, weight_dir: Optional[str]):
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model_Conv1d.pth')

        # Load the pre-trained weights for Encoder and Decoder
        if self.net_type == 'MLP':
            if len(self.genes) == 32738:
                pre_weights = pre_weights[f'Batch_full']
            else:
                pre_weights = pre_weights[f'Batch_{len(self.genes)}']
        elif self.net_type == 'Conv1d':
            pre_weights = pre_weights[f'Batch']

        self.G.net.load_state_dict({k: v for k, v in pre_weights.items()})

        # Initial the memory block with the normal embeddings
        with torch.no_grad():
            self.G.eval()
            sum_t = self.mem_dim/self.batch_size
            t = 0
            while t < sum_t:
                for _, data in enumerate(self.loader):
                    data = data.to(self.device)
                    real_z, _, _ = self.G(data)
                    self.G.Memory.update_mem(real_z)
                    t += 1

    def update_D(self, data):
        self.opt_D.zero_grad()
        _, fake_data, _ = self.G(data)

        real_d = self.D(data)
        self.D_scaler.scale(-torch.mean(real_d)).backward()

        fake_d = self.D(fake_data.detach())
        self.D_scaler.scale(torch.mean(fake_d)).backward()

        # Compute W-div gradient penalty
        gp = calculate_gradient_penalty(data, fake_data, self.D)
        self.D_scaler.scale(gp*self.weight['w_gp']).backward()

        self.D_loss = -torch.mean(real_d) + torch.mean(fake_d) + gp*self.weight['w_gp']

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()
    
    def update_G(self, data):
        real_z, fake_data, fake_z = self.G(data)
        fake_d = self.D(fake_data)

        Loss_enc = self.L2(real_z, fake_z)
        Loss_rec = self.L1(data, fake_data)
        Loss_adv = -torch.mean(fake_d)

        self.G_loss = (self.weight['w_enc']*Loss_enc +
                       self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.opt_G.zero_grad()
        self.G_scaler.scale(self.G_loss).backward(retain_graph=True)
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

        self.G.Memory.update_mem(fake_z)

