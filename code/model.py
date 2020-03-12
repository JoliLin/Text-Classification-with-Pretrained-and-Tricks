#import cupy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FocalLoss import FocalLoss as focal

class MLP(nn.Module):
	def __init__(self, dim=64, classes=2):
		super(MLP, self).__init__()
		self.loss_func = focal()#nn.CrossEntropyLoss()

		emb_size = 10
		self.layer1 = nn.Conv2d(1, emb_size, (1, 300))
	
		self.enc_size = [emb_size]*(2+1)
		
		blocks = [self.linear_block(in_f, out_f) for in_f, out_f in zip(self.enc_size, self.enc_size[1:])]
		self.encoder = nn.Sequential(*blocks)
		self.last = self.linear_block(emb_size, emb_size)

		self.decoder = nn.Linear(emb_size, classes)

		self.IC_input = self.IC(256)

	def IC(self, in_f, *args, **kwargs ):
		return nn.Sequential( nn.BatchNorm1d(in_f), nn.Dropout() )

	def linear_block(self, in_f, out_f, *args, **kwargs ):
		return nn.Sequential(
			self.IC(in_f),
			nn.Linear(in_f, int(in_f/3)),
			nn.ReLU(),
			self.IC(int(in_f/3)),
			nn.Linear(int(in_f/3), out_f),
			nn.ReLU(),
		)

	def linear_last(self, in_f, out_f, *args, **kwargs ):
		return nn.Sequential(
			self.IC(in_f),
			nn.Linear(in_f, int(in_f/3)),
			nn.ReLU(),
			self.IC(int(in_f/3)),
			nn.Linear(int(in_f/3), out_f),
		)
	
	def main_task(self, h):
		h = self.IC_input(h)
		h = self.layer1(h.unsqueeze(1)).squeeze(3)
		
		h = F.relu(h)
		h = F.avg_pool2d(h, (1, h.size(2))).squeeze(2)

		h_org = h
		h = self.encoder(h)
		h = self.last(h)
		h += h_org
		h = self.decoder(h)

		return h
	
	def forward(self, data_, mode='train'):
		if mode == 'inference':
			return self.main_task(data_[0])
		else:
			x_ = data_[0]
			y_ = data_[1]
			y_rating = self.main_task(x_)

			return y_rating, self.loss_func(y_rating, y_)
