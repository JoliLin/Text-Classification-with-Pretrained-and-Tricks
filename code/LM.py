from collections import Counter, deque
from itertools import chain, product
import numpy as np
import util

class BOW:
	def __init__( self, corpus ):
		self.corpus = corpus
		#self.LM = self.preprocess( self.corpus )
		self.key_id = None
		self.dim = 0

	def load_LM( self ):
		return self.LM

	def to_array( self, dic ):
		embed = [0] * self.dim

		for k in dic.keys():
			embed[self.key_id.get(k)] = dic[k]
		
		return embed

	def preprocess( self, corpus, mode='TF' ):
		tf = [Counter(c.split(' ')) for c in corpus]

		keys = chain.from_iterable([list(i.keys()) for i in tf])

		self.key_id = {i[1]:i[0] for i in enumerate(set(keys))}
		self.dim = len(self.key_id.keys())
		
		if mode == 'TF':
			embed_corpus = list(map(self.to_array, tf))
		'''
		elif mode == 'TF-IDF':
			idf = Counter(keys)

			embed = [0] * len(self.key_id)
			
		elif mode == 'binary':
			embed = [0] * len(self.key_id)
		'''	
		return embed_corpus

class ONE_hot:
	def load_LM( self ):
		return self.LM

	def preprocess( self, corpus ):
		c_ = [c for c in corpus]
		key_id = list(set(chain.from_iterable(c_)))
		
		return [list(map(key_id.index, k)) for k in c_]

class pretrained:
	def __init__( self ):
		self.model = util.load_word2vec()

	def load_LM( self ):
		return self.LM

	def word2vec( self, w ):
		try:
			return np.array( self.model[w] ) 

			#return( np.array(list(map(util.quantize, self.model[w]))) )
			
		except KeyError:
			return np.array(300*[0])

	def preprocess( self, corpus ):
		embeddings = []
		for c in corpus:
			embeddings.append(list(map(self.word2vec, c)))
		
		
		#embeddings = [list(map(self.word2vec, c)) for c in corpus]


		return embeddings

	def preprocess_SST( self ):
		return [list(map(self.word2vec, c)) for c in self.corpus]

class pretrained_2:
	def __init__( self ):
		self.model = util.load_word2vec()

	def load_LM( self ):
		return self.LM

	def word2vec( self, w ):
		try:
			return np.array( self.model[w] ) 
		except KeyError:
			return np.array([0])

	def preprocess( self, corpus):
		zero = 0
		non_zero = 0
	
		embeddings = [list(map(self.word2vec, c)) for c in corpus]
		
		for i in embeddings:
			for j in range(len(i)):
				if len(i[j]) < 300:	
					lower = j-1 if j-1 > -1 else 0
					higher = j+1 if j+1 < len(i) else len(i)-1
		
					if len(i[lower]) < 300 and len(i[higher]) < 300:
						i[j] = np.array(300*[0])
						zero += 1
					else:			
						i[j] = (i[lower] + i[higher])/2
						non_zero += 1

		print(zero, non_zero)
		
		return embeddings

class stopLM:
	def __init__( self ):
		self.stop = util.getStop()
		self.model = util.load_word2vec()

	def load_LM( self ):
		return self.LM

	def word2vec( self, w ):
		try:
			return np.array( self.model[w] )
		except KeyError:
			return np.array(300*[0])

	def stopped( self, w ):
		return 0 if w in self.stop else 1
		
	def preprocess( self, corpus ):
		embeddings = []
		labels = []
		
		for c in corpus:
			_  = c.split(' ')
			embeddings.append(list(map(self.word2vec, _)))
			labels.append(list(map(self.stopped, _)))
	
		for i in embeddings:
			for j in range(len(i)):
				if len(i[j]) < 300:	
					lower = j-1 if j-1 > -1 else 0
					higher = j+1 if j+1 < len(i) else len(i)-1
		
					if len(i[lower]) < 300 and len(i[higher]) < 300:
						i[j] = np.array(300*[0])
						zero += 1
					else:			
						i[j] = (i[lower] + i[higher])/2
						non_zero += 1
	
		return embeddings, labels

class WordHashing:
	def __init__(self, n=3):
		self.token = self.genToken()
		self.n = n
		print(len(self.token))
	
	def genToken(self):
		c = 'abcdefghijklmnopqrstuvwxyz1234567890'
		p0 = [''.join(i) for i in product(c, c)]
		p = [''.join(i) for i in product(p0, c)]
		s = ['#{}#'.format(i) for i in c]

		e1 = ['#'+i for i in p0]
		e2 = [i+'#' for i in p0]
	
		symbol = ['#.#', '#,#']
		return ['# #']+[i for i in set(symbol+p+s+e1+e2)]

	def trans2vec( self, word, dim ):
		vec = dim*[0]
		for i in word:
			vec[i] = 1
	
		return vec

	def wordHashing( self, sent ):
		#a sentence is being processed after tokenized
		hashingDoc = []
		hashing = []
		bow = {}
		n = self.n

		for s in sent:	
			s = '#{}#'.format(s)
			i = 0
			while i+n < len(s)+1:
				try:
					hashing.append(self.token.index(s[i:i+n]))
				except ValueError:
					hashing.append(self.token.index('# #'))	
				
				i+=1
			
			'''
			if len(s) > n-2:
				tokens =  [s[i:i+n] for i in range(0, len(s)-1)]	
				tokens[-1] = tokens[-1]+'#'		
				head =  ''.join(['#',tokens[0][:n-1]])
				tokens.insert(0, head)
				hashingDoc.extend( tokens )	
				bow[s] = tokens
			else:
				tokens = [''.join(['#',s,'#'])]
				hashingDoc.extend( tokens )	
				bow[s] = tokens
			'''
		'''
		hashing = [] 
		for t in hashingDoc:
			try:
				hashing.append(self.token.index(t))
			except ValueError:
				hashing.append(self.token.index('# #'))
		'''

		return hashing

	def preprocess( self, corpus ):
		#return [list(map(self.wordHashing, c)) for c in corpus]
		output = []
		for c in corpus:
			output.append(self.wordHashing(c))

		return output
