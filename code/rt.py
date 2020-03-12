import LM
import nltk
import numpy as np
import util
import tokenizer
import torch
from sklearn.model_selection import train_test_split

def load_data( lst ):
	x = []
	y = []

	for i in lst:
		y_, x_ = i.split(' ', 1)

		'''
		x_nltk = util.normalizeString(x_)
		x_nltk = nltk.word_tokenize(x_nltk)
		x_nltk = util.removeStop(x_nltk)
		#x_nltk = ' '.join(x_nltk)
		x_ = x_nltk
		'''

		y.append(int(y_))
		x.append(x_)

	return x, y		

class rt:
	def __init__(self, path = '../data/rt-polarity.all'):
		data = open(path, encoding='utf-8', errors='ignore').readlines()

		np.random.seed(0)
		np.random.shuffle(data)

		wv = LM.pretrained_2()

		x, y = load_data(data)
		x = map(tokenizer.tokenize, x)
		x = map(tokenizer.remove_stopwords, x)

		x = wv.preprocess(list(x))

		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
		train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.1)

		max_len=256
		padding_item = [300*[0]]

		train_x = util.padding(train_x, max_len=max_len, padding_item=padding_item)
		dev_x = util.padding(dev_x, max_len=max_len, padding_item=padding_item)
		test_x = util.padding(test_x, max_len=max_len, padding_item=padding_item)
		print(np.array(train_x).shape)
		print(np.array(dev_x).shape)
		print(np.array(test_x).shape)

		self.data = (train_x, train_y), (dev_x, dev_y), (test_x, test_y) 

		train_type = [torch.FloatTensor, torch.LongTensor]
		val_type = [torch.FloatTensor, torch.LongTensor]
		test_type = [torch.FloatTensor, torch.LongTensor]
		self.data_type = [train_type, val_type, test_type]

		#self.data = (test_x, test_y)
		#self.data_type = test_type
