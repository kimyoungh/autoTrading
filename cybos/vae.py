"""
	Module for multifactor stock embedding
	
	@author: Younghyun Kim
	@Edited: 2020.03.04
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def reparametrize(mu, logvar):
	"""
		function for reparametrization trick
	"""

	std = logvar.div(2).exp()
	eps = Variable(std.data.new(std.size()).normal_())

	return mu + std*eps

class View(nn.Module):
	" view "
	def __init__(self, size):
		super(View, self).__init__()
		self.size = size

	def forward(self, tensor):
		return tensor.view(self.size)

class BetaVae(nn.Module):
	"""
		Model based on original
		beta-VAE paper(Higgins et al, ICLR, 2017)
		for multifactor stock embedding
	"""

	def __init__(self, input_dim, z_dim=8):
		super(BetaVae, self).__init__()
		self.input_dim = input_dim
		self.z_dim = z_dim

		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 32),
			nn.ReLU(True),
			nn.Linear(32, 32),
			nn.ReLU(True),
			nn.Linear(32, z_dim*2),
			)

		self.decoder = nn.Sequential(
			nn.Linear(z_dim, 32),
			nn.ReLU(True),
			nn.Linear(32, 32),
			nn.ReLU(True),
			nn.Linear(32, input_dim)
			)

		self.weight_init()

	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				kaiming_init(m)

	def forward(self, x):
		distributions = self._encode(x)
		mu = distributions[:, :self.z_dim]
		logvar = distributions[:, self.z_dim:]
		z = reparametrize(mu, logvar)
		x_recon = self._decode(z)

		return x_recon, mu, logvar

	def _encode(self, x):
		return self.encoder(x)

	def _decode(self, z):
		return self.decoder(z)

def kaiming_init(m):
	init.kaiming_normal(m.weight)
	if m.bias is not None:
		m.bias.data.fill_(0)
