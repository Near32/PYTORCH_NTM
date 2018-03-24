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


from models import betaVAE_NTM, Bernoulli
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
	path = 'Omniglot--img{}-lr{}-conv{}-out{}'.format(img_dim,lr,conv_dim,args.nbr_character)
	
	if not os.path.exists( './data/{}/'.format(path) ) :
		os.mkdir('./data/{}/'.format(path))
	if not os.path.exists( './data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./data/{}/reconst_images/'.format(path))
	if not os.path.exists( './data/{}/gen_images/'.format(path) ) :
			os.mkdir('./data/{}/gen_images/'.format(path))
	
	
	SAVE_PATH = './data/{}/'.format(path) 

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
	
	if args.trainVAE :
		data_loader = torch.utils.data.DataLoader(dataset=dataset,
													batch_size=batch_size, 
													shuffle=True)
		train_VAE(betavae=betaVAENTM.betaVAE, data_loader=data_loader, optimizer=optimizer, path=path,SAVE_PATH=SAVE_PATH, nbr_epoch=args.epoch,batch_size=args.batch_size,offset=args.offset)
		
	if args.query :
		data_loader = torch.utils.data.DataLoader(dataset=dataset,
													batch_size=batch_size, 
													shuffle=True)
		query(betaVAENTM,data_loader, path, args)

		

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

			cum_latent_acc = 0.0
			iteration_latent = 0
				

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
				
				output = betaVAENTM.ext_output[-1]
				# Accuracy :
				acc = (output.cpu().data.max(1)[1] == label.cpu().data.max(1)[1])
				acc = acc.numpy().mean()*100.0
				cum_latent_acc = (cum_latent_acc*iteration_latent + acc)/(iteration_latent+1)
				iteration_latent += 1
				print('Accuracy : {} %.'.format(cum_latent_acc))

				var_task_loss = var_task_loss + total_loss
				epoch_loss += total_loss.cpu().data[0]

				# VAE : Backprop + Optimize :
				optimizer.zero_grad()
				VAE_loss.backward(retain_graph=True)
				optimizer.step()
				

				if idx_sample % 10 == 0:
				    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
				           "Reconst Loss: %.4f // KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
				           %(epoch+1, nbr_epoch, idx_sample+1, nbrSample4Task, total_loss.data[0], 
				             VAE_loss.data[0], betaVAENTM.kl_divergence.data[0], betaVAENTM.expected_log_lik.exp().data[0]) )
				    if best_loss is not None :
				    	print("Epoch Loss : {} / Best : {}".format(epoch_loss, best_loss))

			# Backprop + Optimize :
			optimizer.zero_grad()
			#var_task_loss.backward(retain_graph=True)
			var_task_loss.backward()
			optimizer.step()
			
			# Temporary save :
			lp = os.path.join(SAVE_PATH,'temp')
			betaVAENTM.save(path=lp)
		
			

		if best_loss is None :
			#first validation : let us set the initialization but not save it :
			best_loss = epoch_loss		
		
		if epoch_loss < best_loss:
			best_loss = epoch_loss
			lp = os.path.join(SAVE_PATH,'best')
			betaVAENTM.save(path=lp)


def train_VAE(betavae,data_loader, optimizer,path,SAVE_PATH,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	z_dim = betavae.z_dim
	img_depth=betavae.img_depth
	img_dim = betavae.img_dim

	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['label']
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	if not stacking :
		torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images.png'.format(path))
	else :
		fixed_x0 = fixed_x.view( (-1, 1, img_depth*img_dim, img_dim) )
		torchvision.utils.save_image(fixed_x0, './data/{}/real_images.png'.format(path))


	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	best_loss = None
	best_model_wts = betavae.state_dict()
	
	for epoch in range(nbr_epoch):
		
		# Save generated variable images :
		nbr_steps = args.querySTEPS
		#mu_mean /= batch_size
		#sigma_mean /= batch_size
		mu_mean = betavae.encode(fixed_x)[1].cpu().data[0]
		sigma_mean = 3.0*torch.ones((z_dim))
		gen_images = torch.ones( (nbr_steps, img_depth, img_dim, img_dim) )
		if stacking :
			gen_images = torch.ones( (nbr_steps, 1, img_depth*img_dim, img_dim) )
			
		for latent in range(z_dim) :
			#var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
			var_z0 = torch.zeros(nbr_steps, z_dim)
			val = mu_mean[latent]-sigma_mean[latent]
			step = 2.0*sigma_mean[latent]/nbr_steps
			print(latent,mu_mean[latent]-sigma_mean[latent],mu_mean[latent],mu_mean[latent]+sigma_mean[latent])
			for i in range(nbr_steps) :
				var_z0[i] = mu_mean
				var_z0[i][latent] = val
				val += step

			var_z0 = Variable(var_z0)
			if use_cuda :
				var_z0 = var_z0.cuda()


			gen_images_latent = betavae.decoder(var_z0)
			gen_images_latent = gen_images_latent.view(-1, img_depth, img_dim, img_dim).cpu().data
			if stacking :
				gen_images_latent = gen_images_latent.view( -1, 1, img_depth*img_dim, img_dim)
			if latent == 0 :
				gen_images = gen_images_latent
			else :
				gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

		#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
		torchvision.utils.save_image(gen_images,'./data/{}/gen_images/{}.png'.format(path,(epoch+offset+1)) )

		mu_mean = 0.0
		sigma_mean = 0.0

		epoch_loss = 0.0
		

		for i, sample in enumerate(data_loader):
			images = sample['image'].float()
			# Save the reconstructed images
			if i % 100 == 0 :
				reconst_images, _, _ = betavae(fixed_x)
				reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
				orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
				ri = torch.cat( [orimg, reconst_images], dim=2)
				if stacking :
					ri = reconst_images.view( (-1, 1, img_depth*img_dim, img_dim) )
				torchvision.utils.save_image(ri,'./data/{}/reconst_images/{}.png'.format(path,(epoch+offset+1) ) )
				
				betavae.save(SAVE_PATH+'temp')
				print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'temp')) )

			images = Variable( (images.view(-1, img_depth,img_dim, img_dim) ) )#.float()
			
			if use_cuda :
				images = images.cuda() 

			out, mu, log_var = betavae(images)
			
			mu_mean += torch.mean(mu.data,dim=0)
			sigma_mean += torch.mean( torch.sqrt( torch.exp(log_var.data) ), dim=0 )

			# Compute :
			#reconstruction loss :
			reconst_loss = F.binary_cross_entropy( out, images, size_average=False)
			#reconst_loss = nn.MultiLabelSoftMarginLoss()(input=out_logits, target=images)
			#reconst_loss = F.binary_cross_entropy_with_logits( input=out, target=images, size_average=False)
			#reconst_loss = F.binary_cross_entropy( Bernoulli(out).sample(), images, size_average=False)
			#reconst_loss = torch.mean( (out.view(-1) - images.view(-1))**2 )
			
			# expected log likelyhood :
			try :
				#expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
				expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )
			except Exception as e :
				print(e)
				expected_log_lik = Variable(torch.ones(1).cuda())
			
			# kl divergence :
			kl_divergence = 0.5 * torch.mean( torch.sum( (mu**2 + torch.exp(log_var) - log_var -1), dim=1) )
			#kl_divergence = 0.5 * torch.sum( (mu**2 + torch.exp(log_var) - log_var -1) )

			# ELBO :
			elbo = expected_log_lik - betavae.beta * kl_divergence
			
			# TOTAL LOSS :
			total_loss = reconst_loss + betavae.beta*kl_divergence
			#total_loss = reconst_loss
			#total_loss = -elbo

			# Backprop + Optimize :
			optimizer.zero_grad()
			total_loss.backward()

			#--------------------------
			#betavae.encoder.localization.zero_grad()
			#betavae.encoder.fc_loc.zero_grad()
			#nn.utils.clip_grad_norm( betavae.encoder.localization.parameters(), args.clip)
			#nn.utils.clip_grad_norm( betavae.encoder.fc_loc.parameters(), args.clip)
			#--------------------------
			
			optimizer.step()

			del images
			
			epoch_loss += total_loss.cpu().data[0]

			if i % 10 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, nbr_epoch, i+1, iter_per_epoch, total_loss.data[0], 
			             reconst_loss.data[0], kl_divergence.data[0],expected_log_lik.exp().data[0]) )

		if best_loss is None :
			#first validation : let us set the initialization but not save it :
			best_loss = epoch_loss
			best_model_wts = betavae.state_dict()
			# save best model weights :
			betavae.save(SAVE_PATH+'best')
			print('Model saved at : {}'.format(SAVE_PATH+'best') )
		elif epoch_loss < best_loss:
			best_loss = epoch_loss
			best_model_wts = betavae.state_dict()
			# save best model weights :
			betavae.save(SAVE_PATH+'best')
			print('Model saved at : {}'.format(SAVE_PATH+'best') )


def query(model,data_loader,path,args):
	global use_cuda

	z_dim = model.betaVAE.z_dim
	img_depth=model.betaVAE.img_depth
	img_dim = model.betaVAE.img_dim
	
	data_iter = iter(data_loader)
	
	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x = sample['image']

	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images_query.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	# variations over the latent variable :
	#sigma_mean = args.queryVAR*torch.ones((z_dim))
	#mu_mean = torch.zeros((z_dim))
	z, mu, log_sig_sq  = model.betaVAE.encode(fixed_x)
	mu_mean = mu.cpu().data[0]#.unsqueeze(0)
	sigma_mean = torch.exp(log_sig_sq).sqrt().cpu().data[0]#.unsqueeze(0)
	print(z,mu_mean,sigma_mean)
	#print(torch.cat([z[0],mu_mean,sigma_mean],dim=1) )

	# Save generated variable images :
	nbr_steps = 16
	gen_images = torch.ones( (nbr_steps, img_depth, img_dim, img_dim) )

	for latent in range(z_dim) :
		#var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
		var_z0 = torch.zeros(nbr_steps, z_dim)
		val = mu_mean[latent]-sigma_mean[latent]
		step = 2.0*sigma_mean[latent]/nbr_steps
		print(latent,mu_mean[latent]-sigma_mean[latent],mu_mean[latent],mu_mean[latent]+sigma_mean[latent])
		for i in range(nbr_steps) :
			var_z0[i] = mu_mean
			var_z0[i][latent] = val
			val += step

		var_z0 = Variable(var_z0)
		if use_cuda :
			var_z0 = var_z0.cuda()

		print(var_z0)
		gen_images_latent = model.betaVAE.decoder(var_z0)
		gen_images_latent = gen_images_latent.view(-1, img_depth, img_dim, img_dim).cpu().data
		if latent == 0 :
			gen_images = gen_images_latent
		else :
			gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

	#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
	torchvision.utils.save_image(gen_images,'./data/{}/gen_images/query.png'.format(path) )


	reconst_images, _, _ = model.betaVAE(fixed_x)
	reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
	orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
	ri = torch.cat( [orimg, reconst_images], dim=2)
	torchvision.utils.save_image(ri,'./data/{}/reconst_images/query.png'.format(path ) )
		





if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser(description='Neural Turing Machine - Omniglot')
	parser.add_argument('--train',action='store_true',default=False)
	parser.add_argument('--trainVAE',action='store_true',default=False)
	parser.add_argument('--query',action='store_true',default=False)
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
	parser.add_argument('--queryVAR', type=float, default=3.0)
	parser.add_argument('--querySTEPS', type=int, default=8)
	parser.add_argument('--clip', type=float, default=1e-5)
	
	args = parser.parse_args()
	print(args)

	setting(args)