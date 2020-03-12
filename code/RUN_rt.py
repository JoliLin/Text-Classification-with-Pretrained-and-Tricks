import args

import numpy as np
import model as models
import training_handler as trainer
import util
from rt import rt
import torch.utils.data as Data
from torch.autograd import Variable

def inference():
	thandler = trainer.handler(args.process_command())

	rt_data = rt()
	data = trainer.load_data(rt_data.data, data_type=rt_data.data_type)
	test_loader = data

	model_ = models.MLP(300, classes=2)
	#print(model_)
	total = sum(p.numel() for p in model_.parameters() if p.requires_grad)
	print('# of para: {}'.format(total))	

	model_name = 'MLP.pt'

	predicted = thandler.predict( model_, test_loader, model_name)	

	print([np.argmax(np.array(i)) for i in predicted])

#inference()
	

if __name__ == '__main__':
#def test():
	scores = []

	thandler = trainer.handler(args.process_command())

	### load data
	rt_data = rt()
	data = trainer.data_loader_with_dev(rt_data.data, data_type=rt_data.data_type)
	train_loader, valid_loader, test_loader = data

	for i in range(1):
		### load model
		model_ = models.MLP(300, classes=2)
		#print(model_)
		total = sum(p.numel() for p in model_.parameters() if p.requires_grad)
		print('# of para: {}'.format(total))	

		model_name = 'MLP.pt'
	
		### train & test
		thandler.train( model_, train_loader, valid_loader, model_name )
		y_true, y_pred, avg_loss = thandler.test( model_, test_loader, model_name )
	
		score = util.PRF([i.detach().cpu() for i in y_true], [i.detach().cpu() for i in y_pred])
		scores.append(score)

		print(scores)
		print(sum(scores)/len(scores))
