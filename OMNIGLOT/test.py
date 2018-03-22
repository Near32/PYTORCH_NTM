import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image 
from omniglot import Omniglot

def test_Omniglot() :
	import cv2
	
	root = './data/'
	dataset = Omniglot(root=root,download=True)
	#dataset = Omniglot(root=root,background=False,download=True)
	print(len(dataset))
	idx = 0
	
	idx_alphabet = 0
	idx_character = 0
	idx_sample = 0 

	while True :
		
		sample = dataset[idx]
		image_path = dataset.sampleSample4Character4Alphabet( idx_alphabet, idx_character, idx_sample)
		
		img = np.array( sample[0]) 
		cv2.imshow('test',img)
		
		key = cv2.waitKey(1) & 0xFF
		if  key == ord('q'):
			break
		elif key == ord('n') :
			idx += 1
			#print(sample[1])
			print(image_path)
			idx_sample+=1
			print("next image")


def test_OmniglotClass() :
	import cv2
	
	root = './data/'
	dataset = Omniglot(root=root,download=True)
	#dataset = Omniglot(root=root,background=False,download=True)
	print(len(dataset))
	idx = 0
	
	idx_alphabet = 0
	idx_character = 0
	idx_sample = 0 

	sample = dataset.getSample( idx_alphabet, idx_character, idx_sample)
	image_path = dataset.sampleSample4Character4Alphabet( idx_alphabet, idx_character, idx_sample)
	changed = False
	
	while True :
		
		#sample = dataset[idx]
		
		img = np.array( sample[0]) 
		cv2.imshow('test',img)
		
		key = cv2.waitKey(1) & 0xFF
		if  key == ord('q'):
			break
		elif key == ord('n') :
			idx += 1
			print(image_path)
			idx_sample+=1
			changed = True
		elif key == ord('a') :
			idx += 1
			print(image_path)
			idx_alphabet+=1
			idx_character = 0 
			idx_sample = 0
			changed = True
		elif key == ord('c') :
			idx += 1
			print(image_path)
			idx_character+=1
			idx_sample = 0 
			changed = True
			
		if changed :
			changed = False
			sample = dataset.getSample( idx_alphabet, idx_character, idx_sample)
			image_path = dataset.sampleSample4Character4Alphabet( idx_alphabet, idx_character, idx_sample)
			


def test_OmniglotTask() :
	import cv2
	
	root = './data/'
	dataset = Omniglot(root=root,download=True)
	#dataset = Omniglot(root=root,background=False,download=True)
	print(len(dataset))
	
	idx_alphabet = 0
	idx_sample = 0 

	#task, nbrCharacter4Task, nbrSample4Task = dataset.generateFewShotLearningTask( alphabet_idx=idx_alphabet) 
	task, nbrCharacter4Task, nbrSample4Task = dataset.generateIterFewShotLearningTask( alphabet_idx=idx_alphabet) 
	sample = dataset.getSample( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'] )
	image_path = dataset.sampleSample4Character4Alphabet( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'])
	changed = False
	taskChanged = False
	idx = 0
	while True :
		
		#sample = dataset[idx]
		
		img = np.array( sample[0]) 
		cv2.imshow('test',img)
		
		key = cv2.waitKey(1) & 0xFF
		if  key == ord('q'):
			break
		elif key == ord('n') :
			idx += 1
			idx_sample= (idx_sample+1) % len(task)
			changed = True
		elif key == ord('a') :
			idx_alphabet+=1
			idx_character = 0 
			idx_sample = 0
			taskChanged = True
			
		if changed :
			changed = False
			sample = dataset.getSample( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'] )
			image_path = dataset.sampleSample4Character4Alphabet( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'])
			print(image_path,'/{}'.format(nbrCharacter4Task))
			print('Sample {} / {}'.format(idx, nbrSample4Task))

		if taskChanged :
			taskChanged = False
			idx = 0
			task, nbrCharacter4Task, nbrSample4Task = dataset.generateFewShotLearningTask( alphabet_idx=idx_alphabet) 
			sample = dataset.getSample( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'] )
			image_path = dataset.sampleSample4Character4Alphabet( task[idx_sample]['alphabet'], task[idx_sample]['character'], task[idx_sample]['sample'])
			print(image_path,'/{}'.format(nbrCharacter4Task))


def test_OmniglotSeq() :
	import cv2
	
	root = './data/'
	h = 240
	w = 240
	dataset = Omniglot(root=root,h=h,w=w)
	#dataset = Omniglot(root=root,background=False,download=True)
	print(len(dataset))
	
	idx_alphabet = 0
	idx_sample = 0 

	seq, nbrCharacter4Task, nbrSample4Task = dataset.generateIterFewShotInputSequence( alphabet_idx=idx_alphabet) 
	sample = dataset.getSample( seq[idx_sample]['alphabet'], seq[idx_sample]['character'], seq[idx_sample]['sample'] )
	image_path = dataset.sampleSample4Character4Alphabet( seq[idx_sample]['alphabet'], seq[idx_sample]['character'], seq[idx_sample]['sample'])
	changed = False
	seqChanged = False
	idx = 0
	while True :
		
		#sample = dataset[idx]
		
		img = (sample['image'].numpy()*255).transpose( (1,2,0) ) 
		cv2.imshow('test',img)
		
		key = cv2.waitKey(1) & 0xFF
		if  key == ord('q'):
			break
		elif key == ord('n') :
			idx += 1
			idx_sample= (idx_sample+1) % len(seq)
			changed = True
		elif key == ord('a') :
			idx_alphabet+=1
			idx_character = 0 
			idx_sample = 0
			seqChanged = True
			
		if changed :
			changed = False
			sample = dataset.getSample( seq[idx_sample]['alphabet'], seq[idx_sample]['character'], seq[idx_sample]['sample'] )
			image_path = dataset.sampleSample4Character4Alphabet( seq[idx_sample]['alphabet'], seq[idx_sample]['character'], seq[idx_sample]['sample'])
			print(image_path,'/{}'.format(nbrCharacter4Task))
			print('Sample {} / {}'.format(idx, nbrSample4Task))
			print('Target :{}'.format(seq[idx_sample]['target']))

		if seqChanged :
			seqChanged = False
			idx = 0
			seq, nbrCharacter4Task, nbrSample4Task = dataset.generateFewShotLearningTask( alphabet_idx=idx_alphabet) 
			sample = dataset.getSample( seq[idx_sample]['alphabet'], seq[idx_sample]['character'], seq[idx_sample]['sample'] )
			image_path = dataset.sampleSample4Character4Alphabet( seq[idx_sample]['alphabet'], seq[idx_sample]['character'], seq[idx_sample]['sample'])
			print(image_path,'/{}'.format(nbrCharacter4Task))

from collections import namedtuple

Item = namedtuple('Item', ('alphabet','character','sample','target','nbrCharacter') )

def test_OmniglotBatchedSeq() :
	import cv2
	
	root = './data/'
	h = 240
	w = 240
	dataset = Omniglot(root=root,h=h,w=w)
	#dataset = Omniglot(root=root,background=False,download=True)
	print(len(dataset))
	
	batch_size = 2
	max_nbr_char = 5
	idx_alphabet = list()
	for i in range(batch_size) : idx_alphabet.append(i)

	idx_sample = 0 

	keys = ('alphabet','character','sample','target','nbrCharacter')
	seqs = list()
	for s in range(batch_size) :
		seq, nbrCharacter4Task, nbrSample4Task = dataset.generateIterFewShotInputSequence( alphabet_idx=idx_alphabet[s],max_nbr_char=max_nbr_char)
		dseqs = {'alphabet':[],'character':[],'sample':[],'target':[],'nbrCharacter':[] }
		for el in seq :
			for k in el.keys() :
				dseqs[k].append( el[k] )
		seqs.append(dseqs)
	seqs = [ {k:[seqs[b][k] for b in range(batch_size)] }for k in keys]
	dseqs = dict()
	for d in seqs :
		dseqs.update(d)
	# key x task_idx x seq_idx
	for k in dseqs.keys() :
		dseqs[k] = zip(*dseqs[k])
	print(seqs)
	raise
	batch = dataset.getBatchedSample(seqs)


if __name__ == "__main__" :
	#test_Omniglot()
	#test_OmniglotClass()
	#test_OmniglotTask()
	test_OmniglotSeq()
	#test_OmniglotBatchedSeq()

