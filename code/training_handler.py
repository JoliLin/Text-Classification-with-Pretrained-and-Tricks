import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

import numpy as np
import time
import FGM
from earlyStopping import EarlyStopping

def load_data( data, data_type=[torch.LongTensor, torch.LongTensor], batch_size=64 ):

	torch_dataset = Data.TensorDataset( *[Variable(i[1](i[0])) for i in zip(data, data_type)] )
			
	data_loader = Data.DataLoader( dataset=torch_dataset, batch_size=batch_size )
			
	return data_loader

def data_loader_with_dev( data_, data_type=[[], [], []] ):
	return tuple([load_data(*i) for i in zip(data_, data_type)])

class handler:
	def __init__(self, arg):
		self.gpu = arg.gpu
		self.epoch_size = arg.epoch
		self.batch_size = arg.batch
		self.device = self.device_setting( self.gpu )
		self.hist = dict()

	def device_setting( self, gpu=1 ):
		return 'cpu' if gpu == -1 else 'cuda:{}'.format(gpu)

	def print_hist( self ):
		string = '===> '

		for k in self.hist.keys():
			val = self.hist[k]

			if type(val) == float:
				string += '{}:{:.6f}\t'.format(k,val)	
			else:
				string += '{}:{}\t'.format(k,val)
		print(string)	
			
	def train( self, model_, train_loader, valid_loader, model_name='checkpoint.pt' ):
		es = EarlyStopping(patience=15, verbose=True)

		optimizer = torch.optim.Adam(model_.parameters())
		#optimizer = torch.optim.SGD(model_.parameters(), lr=3e-4, momentum=0.9)	
		model_.to(self.device)

		start = time.time()
		best_model = None
	
		for epoch in range(self.epoch_size):
			start = time.time()
		
			model_.train()
			train_loss, valid_loss = 0.0, 0.0
			train_acc = 0.0
			N_train = 0

			fgm = FGM.FGM(model_)

			for i, data_ in enumerate(train_loader):
				data_ = [_.to(self.device) for _ in data_]
				optimizer.zero_grad()
				y_pred, loss = model_(data_, mode='train')
				
				loss.backward()
				###attack
				fgm.attack()
				y_pred ,loss_atk = model_(data_, mode='train')
				loss_atk.backward()
				fgm.restore()
				###
				
				train_loss += loss.item()*len(data_[0])
				train_acc += self.accuracy( y_pred, data_[1] ).item()*len(data_[0])
				optimizer.step()
				N_train += len(data_[0])

			self.hist['Epoch'] = epoch+1
			self.hist['time'] = time.time()-start
			#self.hist['train_loss'] = train_loss/len(train_loader.dataset)		
			#self.hist['train_acc'] = train_acc/len(train_loader.dataset)
			self.hist['train_loss'] = train_loss/N_train	
			self.hist['train_acc'] = train_acc/N_train

			torch.save(model_.state_dict(), model_name)
			
			if valid_loader != None:
				valid_true, valid_pred, valid_loss = self.test(model_, valid_loader, model_name)

				es(valid_loss, model_, model_name)
				
			self.print_hist()
			
			if es.early_stop:	
				print('Early stopping')
				break
				
	def test( self, model_, test_loader, model_name='checkpoint.pt' ):
		model_.load_state_dict(torch.load(model_name))
		model_.to(self.device)
		model_.eval()
		test_loss = 0.0
		test_acc = 0.0

		y_pred, y_true = [], []
		N_test = 0
		with torch.no_grad():
			for i, data_ in enumerate(test_loader):
				data_ = [_.to(self.device) for _ in data_]
				logit, loss = model_(data_, mode='test')

				y_pred.extend(logit)
				y_true.extend(data_[1])
				test_loss += loss.item()*len(data_[0])
				test_acc += self.accuracy( logit, data_[1]).item()*len(data_[0])
				N_test += len(data_[0])

		self.hist['val_loss'] = test_loss/N_test
		self.hist['val_acc'] = test_acc/N_test
		
		return y_true , y_pred, test_loss/N_test

	def predict( self, model_, test_loader, model_name='checkpoint.pt' ):
		model_.to(self.device)
		
		model_.load_state_dict(torch.load(model_name))
		model_.eval()

		y_pred = []
		with torch.no_grad():
			for i, data_ in enumerate(test_loader):
				data_ = [_.to(self.device) for _ in data_]
				logit = model_(data_, mode='inference')
				y_pred.extend(logit.detach().cpu())

		return y_pred

	def accuracy( self, y_pred, p_true ):
		return (np.array(list(map(np.argmax, y_pred.detach().cpu()))) == np.array(p_true.cpu())).sum()/len(y_pred)
