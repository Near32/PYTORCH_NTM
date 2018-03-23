import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms 

from skimage import io, transform
from scipy.io import loadmat
from scipy.misc import imresize, imsave

import numpy as np

import math
from PIL import Image


from models import betaVAE_NTM
from OMNIGLOT import Omniglot

use_cuda = True


def setting(args) :
	size = args.imgSize
	batch_size = args.batch_size
	lr = args.lr
	epoch = args.epoch
	
	# Dataset :
	root = './OMNIGLOT/data/'
	h = args.imgSize
	w = args.imgSize
	dataset = Omniglot(root=root,h=h,w=w)
	
	# Model :
	latent_dim = args.latent_dim
	NTMhidden_dim= args.NTMhidden_dim
	NTMoutput_dim= args.nbr_character
	NTMnbr_layers=1
	NTMmem_nbr_slots=128
	NTMmem_dim= 32

	NTMnbr_read_heads=1
	NTMnbr_write_heads=1
	batch_size=args.batch_size
	
	beta=args.beta
	net_depth=4
	img_dim=args.imgSize
	conv_dim=args.conv_dim
	global use_cuda
	img_depth=3

	betaVAENTM = betaVAE_NTM(latent_dim, NTMhidden_dim=NTMhidden_dim, NTMoutput_dim=NTMoutput_dim, NTMnbr_layers=NTMnbr_layers, 
							NTMmem_nbr_slots=NTMmem_nbr_slots, NTMmem_dim= NTMmem_dim, NTMnbr_read_heads=NTMnbr_read_heads, 
							NTMnbr_write_heads=NTMnbr_write_heads, batch_size=batch_size,
							beta=beta,net_depth=net_depth,img_dim=img_dim, conv_dim=conv_dim, use_cuda=use_cuda, img_depth=img_depth)
	frompath = True
	print(betaVAENTM)
		
	# LOADING :
	path = 'Omniglot--img{}-lr{}-conv{}'.format(img_dim,lr,conv_dim)
	
	if not os.path.exists( './data/{}/'.format(path) ) :
		os.mkdir('./data/{}/'.format(path))
	if not os.path.exists( './data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./data/{}/reconst_images/'.format(path))
	
	
	SAVE_PATH = './data/{}'.format(path) 

	if frompath :
		try :
			lp =os.path.join(SAVE_PATH,'best')
			betaVAENTM.load(path=lp) 
			print('NET LOADING : OK.')
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )
			try :
				lp = os.path.join(SAVE_PATH,'temp')
				betaVAENTM.load(path=lp) 
				print('temporary NET LOADING : OK.')
			except Exception as e :
				print('EXCEPTION : temporary NET LOADING : {}'.format(e) )
				

	# OPTIMIZER :
	optimizer = torch.optim.Adam( betaVAENTM.parameters(), lr=lr)
	#optimizer = torch.optim.Adagrad( betaVAENTM.parameters(), lr=lr)
	
	if args.train :
		train_model(betaVAENTM,dataset, optimizer, SAVE_PATH, path, args,nbr_epoch=args.epoch,batch_size=args.batch_size,offset=args.offset)
	

def train_model(betaVAENTM,dataset, optimizer, SAVE_PATH,path,args,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	best_loss = None
	best_model_wts = betaVAENTM.state_dict()
	
	for epoch in range(nbr_epoch):	
		epoch_loss = 0.0
		
		nbr_task_alphabet = dataset.nbrAlphabets()
		
		
		for task_alphabet_idx in range(nbr_task_alphabet) :
			betaVAENTM.reset()
			
			task, nbrCharacter4Task, nbrSample4Task = dataset.generateIterFewShotInputSequence( alphabet_idx=task_alphabet_idx,max_nbr_char=args.nbr_character)

			var_task_loss = 0.0

			prev_label = torch.zeros(nbrCharacter4Task).float()
			prev_label[np.random.randint(nbrCharacter4Task)] = 1 


			for idx_sample in range(nbrSample4Task):
				sample = dataset.getSample( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'], nbrChar=args.nbr_character )
				
				image = sample['image'].float()
				target = prev_label#sample['target'].float()
				label = sample['label'].float()
				prev_label = label 

				image = Variable( image, volatile=False ).unsqueeze(0)
				target = Variable( target, volatile=False ).unsqueeze(0)
				label = Variable( label, volatile=False ).unsqueeze(0)

				if use_cuda :
					image = image.cuda() 
					label = label.cuda() 
					target = target.cuda() 

				total_loss, VAE_loss, NTM_loss = betaVAENTM.compute_losses(x=image,y=label,target=target)
				
				var_task_loss = var_task_loss + total_loss
				epoch_loss += total_loss.cpu().data[0]

				if idx_sample % 10 == 0:
				    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
				           "Reconst Loss: %.4f " 
				           %(epoch+1, nbr_epoch, idx_sample+1, nbrSample4Task, total_loss.data[0], 
				             VAE_loss.data[0]) )
				    if best_loss is not None :
				    	print("Epoch Loss : {} / Best : {}".format(epoch_loss, best_loss))

			# Temporary save :
			lp = os.path.join(SAVE_PATH,'temp')
			betaVAENTM.save(path=lp)
		
			# Backprop + Optimize :
			optimizer.zero_grad()
			#var_task_loss.backward(retain_graph=True)
			var_task_loss.backward()
			optimizer.step()


		if best_loss is None :
			#first validation : let us set the initialization but not save it :
			best_loss = epoch_loss		
		
		if epoch_loss < best_loss:
			best_loss = epoch_loss
			lp = os.path.join(SAVE_PATH,'best')
			betaVAENTM.save(path=lp)

		





if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser(description='Neural Turing Machine - Omniglot')
	parser.add_argument('--train',action='store_true',default=False)
	parser.add_argument('--evaluate',action='store_true',default=False)
	parser.add_argument('--offset', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--beta', type=float, default=1.0)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--conv_dim', type=int, default=64)
	parser.add_argument('--latent_dim', type=int, default=10)
	parser.add_argument('--NTMhidden_dim', type=int, default=240)
	parser.add_argument('--nbr_character', type=int, default=10)
	parser.add_argument('--imgSize', default=120, type=int,help='input image size')
	
	args = parser.parse_args()
	print(args)

	setting(args)