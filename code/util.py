import numpy as np
import re
import unicodedata
import sys
import time
from collections import deque
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics

stop = set(stopwords.words('english'))

def getStop():
	return stop

def sent2words( sent ):
	return word_tokenize(sent) 

def normalizeString(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r" : ", ":", string)
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " ( ", string) 
	string = re.sub(r"\)", " ) ", string) 
	string = re.sub(r"\?", " ? ", string) 
	string = re.sub(r"\s{2,}", " ", string)   
	return string.strip().lower()

def removeStop(lst):
	return [word for word in lst if word not in stop]

def key2value( target, dic ):
	return dic[target]

def list2file( file_name, data ):
	with open( file_name, 'w') as f:
		for i in range(len(data)):
			f.write('{}\n'.format(data[i]))

def padding( x, max_len=256, padding_item=[0] ):
	return [i+(max_len-len(i))*padding_item if len(i) < max_len else i[:max_len] for i in x]

def rmse(y_true, y_pred):
	return mse(y_true, y_pred) ** 0.5

def hit_rate(y_true, y_pred):
	hit = 0
	for i in zip(y_true, y_pred):
		#print(i[0], np.argmax(np.array(i[1])))
		if i[0] == np.argmax(np.array(i[1])):
			hit += 1

	return float(hit)/len(y_true)

def PRF(y_true, y_pred):
	y_pred = [np.argmax(np.array(i)) for i in y_pred]
	#for i in zip(y_true, y_pred):
	#	print(i)

	print('accuracy')
	print(metrics.accuracy_score(y_true, y_pred))
	print('binary')
	print(metrics.precision_score(y_true, y_pred))
	print(metrics.recall_score(y_true, y_pred))
	print(metrics.f1_score(y_true, y_pred))
	
	print('micro')
	print(metrics.precision_score(y_true, y_pred, average='micro'))
	print(metrics.recall_score(y_true, y_pred, average='micro'))
	print(metrics.f1_score(y_true, y_pred, average='micro'))

	print('macro')
	print(metrics.precision_score(y_true, y_pred, average='macro'))
	print(metrics.recall_score(y_true, y_pred, average='macro'))
	print(metrics.f1_score(y_true, y_pred, average='macro'))

	#print('CM')
	#tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
	#p = float(tp/(tp+fp))
	#r = float(tp/(tp+fn))
	#f = float(2*p*r/(p+r))

	#print(p)
	#print(r)
	#print(f)

	return metrics.f1_score(y_true, y_pred)
 

def train_test_split( data, test_size=0.2 ):
	x, y = data
	test_len = int(test_size*len(x))

	train_x = x[test_len:]
	train_y = y[test_len:]
	test_x = x[:test_len]
	test_y = y[:test_len]

	return (train_x, train_y), (test_x, test_y)

def load_word2vec():
	from gensim.models import KeyedVectors
	#model = KeyedVectors.load_word2vec_format('/home/zllin/model.bin', binary=True)
	model = KeyedVectors.load_word2vec_format('/Users/joli/word2vec_model/model.bin', binary=True)

	return model 

def quantize( element ):
	if element == 0.0:
		return 0.0
	elif element > 0:
		return 1.0
	elif element < 0:
		return -1.0
	
	#return 1 if element > 0 else 0
