"""
	trainer for BetaVae
	@author: Younghyun Kim
	@Edited: 2020.03.05
"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from vae import BetaVae

def reconstruction_loss(x, x_recon, distribution):
	batch_size = x.size(0)
	assert batch_size != 0

	if distribution == 'bernoulli':
		recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, 
														size_average=False)\
														.div(batch_size)
	elif distribution == 'gaussian':
		x_recon = F.sigmoid(x_recon)
		recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
	else:
		recon_loss = None

	return recon_loss

def kl_divergence(mu, logvar):
	batch_size = mu.size(0)
	assert batch_size != 0

	klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
	total_kld = klds.sum(1).mean(0, True)
	dimension_wise_kld = klds.mean(0)
	mean_kld = klds.mean(1).mean(0, True)

	return total_kld, dimension_wise_kld, mean_kld

class DataGather(object):
	def __init__(self):
		self.data = self.get_empty_data_dict()
	
	def get_empty_data_dict(self):
		return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

class Trainer(object):
    def __init__(self, input_dim=43, z_dim=8,
                 beta=4, gamma=1000,
                 lr=1e-4, beta1=0.9, beta2=0.999,
                 decoder_dist='gaussian',
                 batch_size=128,
                 val_batch_size=64,
                 cuda=True, cu_num=0):
        if cuda and torch.cuda.is_available():
            self.device = "cuda:{}".format(str(cu_num))
        else:
            self.device = "cpu"

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.decoder_dist = decoder_dist

        self.net = BetaVae(input_dim, z_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters(),
                                lr=lr, betas=(beta1, beta2))

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
