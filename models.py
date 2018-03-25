import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from math import floor

class Distribution(object) :
	def sample(self) :
		raise NotImplementedError

	def log_prob(self,values) :
		raise NotImplementedError

class Bernoulli(Distribution) :
	def __init__(self, probs) :
		self.probs = probs

	def sample(self) :
		return torch.bernoulli(self.probs)

	def log_prob(self,values) :
		log_pmf = ( torch.stack( [1-self.probs, self.probs] ) ).log()
		dum = values.unsqueeze(0).long()
		return log_pmf.gather( 0, dum ).squeeze(0)
		
		#logits, value = broadcast_all(self.probs, values)
		#return -F.binary_cross_entropy_with_logits(logits, value, reduce=False)

		#return -F.binary_cross_entropy_with_logits(self.probs, values)

def conv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
	layers = []
	layers.append( nn.Conv2d( sin,sout, k, stride,pad) )
	if batchNorm :
		layers.append( nn.BatchNorm2d( sout) )
	return nn.Sequential( *layers )

def deconv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
	layers = []
	layers.append( nn.ConvTranspose2d( sin,sout, k, stride,pad) )
	if batchNorm :
		layers.append( nn.BatchNorm2d( sout) )
	return nn.Sequential( *layers )

class STNbasedNet(nn.Module):
    def __init__(self, input_dim=256, input_depth=1, nbr_stn=2, stn_stack_input=True):
        super(STNbasedNet, self).__init__()

        self.input_dim = input_dim
        self.input_depth = input_depth
        self.nbr_stn = nbr_stn
        self.stn_stack_input=stn_stack_input

        # Spatial transformer localization-network
        stnloc = []
        dim = self.input_dim
        pad = 0
        stride = 1
        k=7
        stnloc.append( nn.Conv2d(self.input_depth, 8, kernel_size=k, padding=pad, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        k=2
        stride = 2
        stnloc.append( nn.MaxPool2d(k, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        stnloc.append( nn.ReLU(True) )
        k=5
        stride=1
        stnloc.append( nn.Conv2d(8, 16, kernel_size=k, padding=pad, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        k=2
        stride = 2
        stnloc.append( nn.MaxPool2d(k, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        stnloc.append( nn.ReLU(True) )
        self.localization = nn.Sequential( *stnloc)
        
        #print('DIM OUTPUT : {}'.format(dim) )

        # Regressor for the 3 * 2 affine matrixes :
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * (dim**2), 128),
            nn.ReLU(True),
            #nn.Linear(128, self.nbr_stn * 3 * 2)
            nn.Linear(128, self.nbr_stn * 2 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        #self.fc_loc[2].weight.data += torch.rand( self.fc_loc[2].weight.size() ) * 1e-10
        #init_bias = torch.FloatTensor( [1.0, 0, 0.0, 0, 1.0, 0.0]).view((1,-1))
        init_bias = torch.FloatTensor( [0.25, 0, 0.25, 0] ).view((1,-1))
        for i in range(self.nbr_stn-1 ) :
        	#r = torch.rand( (1,6)) * 1e-10
        	r = torch.rand( (1,4)) * 1e-10
        	#ib = torch.FloatTensor( [0.5, 0, 0.0, 0, 0.5, 0.0]).view((1,-1))
        	ib = torch.FloatTensor( [0.5, 0, 0.5, 0]).view((1,-1))
        	#ib += r
        	init_bias = torch.cat( [init_bias, ib], dim=0)
        self.fc_loc[2].bias.data = init_bias.view((-1))

    # Spatial transformer network forward function
    def stn(self, x):
        batch_size = x.size()[0]
        xs = self.localization(x)
        xs = xs.view(batch_size,-1)
        theta = self.fc_loc(xs)
        #theta = theta.view(batch_size,self.nbr_stn, 2, 3)
        theta = theta.view(batch_size,self.nbr_stn, -1).contiguous()

        xd = []
        zeroft = Variable(torch.zeros((batch_size,1) ) ).cuda()
        for i in range(self.nbr_stn) :
            thetad = theta[:,i,:].contiguous()
            thetad = thetad.view((batch_size,-1,1))
            thetad = thetad.contiguous()
            thetad = torch.cat( [ thetad[:,0], zeroft, thetad[:,1], zeroft, thetad[:,2], thetad[:,3] ], dim=1)
            thetad = thetad.view((-1,2,3)).contiguous()
            grid = F.affine_grid(thetad, x.size())
            xd.append( F.grid_sample(x, grid) )

        if self.stn_stack_input :
            xd.append( x)

        xd = torch.cat( xd, dim=1)
        
        return xd

    def forward(self, x):
        batch_size = x.size()[0]
        # transform the input
        x = self.stn(x)

        return x





class Decoder(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(Decoder,self).__init__()
		
		self.net_depth = net_depth
		self.dcs = []
		outd = conv_dim*(2**self.net_depth)
		ind= z_dim
		k = 4
		dim = k
		pad = 1
		stride = 2
		self.fc = deconv( ind, outd, k, stride=1, pad=0, batchNorm=False)
		
		for i in reversed(range(self.net_depth)) :
			ind = outd
			outd = conv_dim*(2**i)
			self.dcs.append( deconv( ind, outd,k,stride=stride,pad=pad) )
			self.dcs.append( nn.LeakyReLU(0.05) )
			dim = k-2*pad + stride*(dim-1)
		self.dcs = nn.Sequential( *self.dcs) 
			
		ind = outd
		self.img_depth=img_depth
		outd = self.img_depth
		outdim = img_dim
		indim = dim
		pad = 0
		stride = 1
		k = outdim +2*pad -stride*(indim-1)
		self.dcout = deconv( ind, outd, k, stride=stride, pad=pad, batchNorm=False)
		
	def decode(self, z) :
		z = z.view( z.size(0), z.size(1), 1, 1)
		out = F.leaky_relu( self.fc(z), 0.05)
		out = F.leaky_relu( self.dcs(out), 0.05)
		out = F.sigmoid( self.dcout(out))
		return out

	def forward(self,z) :
		return self.decode(z)

class Encoder(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(Encoder,self).__init__()
		
		self.net_depth = net_depth
		self.img_depth= img_depth
		self.z_dim = z_dim
		# 224
		self.cv1 = conv( self.img_depth, 96, 11, batchNorm=False)
		# 108/109 = E( (224-11+2*1)/2 ) + 1
		self.d1 = nn.Dropout2d(p=0.8)
		self.cv2 = conv( 96, 256, 5)
		# 53 / 54
		self.d2 = nn.Dropout2d(p=0.8)
		self.cv3 = conv( 256, 384, 3)
		# 27 / 27
		self.d3 = nn.Dropout2d(p=0.5)
		self.cv4 = conv( 384, 64, 1)
		# 15
		self.d4 = nn.Dropout2d(p=0.5)
		self.fc = conv( 64, 64, 4, stride=1,pad=0, batchNorm=False)
		# 12
		#self.fc1 = nn.Linear(64 * (12**2), 128)
		#self.fc1 = nn.Linear(64 * (14**2), 128)
		self.fc1 = nn.Linear(1600, 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, 64)
		self.bn2 = nn.BatchNorm1d(64)
		self.fc3 = nn.Linear(64, self.z_dim)

	def encode(self, x) :
		out = F.leaky_relu( self.cv1(x), 0.15)
		out = self.d1(out)
		out = F.leaky_relu( self.cv2(out), 0.15)
		out = self.d2(out)
		out = F.leaky_relu( self.cv3(out), 0.15)
		out = self.d3(out)
		out = F.leaky_relu( self.cv4(out), 0.15)
		out = self.d4(out)
		out = F.leaky_relu( self.fc(out))
		#print(out.size())
		out = out.view( -1, self.num_flat_features(out) )
		#print(out.size())
		out = F.leaky_relu( self.bn1( self.fc1( out) ), 0.15 )
		out = F.leaky_relu( self.bn2( self.fc2( out) ), 0.15)
		out = F.relu(self.fc3( out) )


		return out


	def forward(self,x) :
		return self.encode(x)

	def num_flat_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features



class betaVAE(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAE,self).__init__()
		self.encoder = Encoder(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = Decoder(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
		self.beta = beta
		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
		#veps = Variable( eps, requires_grad=False)
		if self.use_cuda :
			veps = veps.cuda()
		z = mu + veps * torch.exp( log_var/2 )
		return z

	def forward(self,x) :
		self.h = self.encoder( x)
		self.mu, self.log_var = torch.chunk(self.h, 2, dim=1 )
		self.z = self.reparameterize( self.mu,self.log_var)
		self.out = self.decoder(self.z)

		return self.out, self.mu, self.log_var

	def encode(self,x) :
		self.h = self.encoder( x)
		self.mu, self.log_var = torch.chunk(self.h, 2, dim=1 )
		self.z = self.reparameterize( self.mu,self.log_var)
		
		return self.z, self.mu, self.log_var

	def save(self,path) :
		# Encoder :
		enc_wts = self.encoder.state_dict()
		encpath = path + 'Encoder.weights'
		torch.save( enc_wts, encpath )
		print('Encoder saved at : {}'.format(encpath) )

		# Decoder :
		dec_wts = self.decoder.state_dict()
		decpath = path + 'Decoder.weights'
		torch.save( dec_wts, decpath )
		print('Decoder saved at : {}'.format(decpath) )


	def load(self,path) :
		# Encoder :
		encpath = path + 'Encoder.weights'
		self.encoder.load_state_dict( torch.load( encpath ) )
		print('Encoder loaded from : {}'.format(encpath) )
		
		# Decoder :
		decpath = path + 'Decoder.weights'
		self.decoder.load_state_dict( torch.load( decpath ) )
		print('Decoder loaded from : {}'.format(decpath) )
		


class STNbasedEncoder(STNbasedNet) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32, nbr_stn=2, stn_stack_input=True ) :
		super(STNbasedEncoder,self).__init__(input_dim=img_dim, input_depth=img_depth, nbr_stn=nbr_stn, stn_stack_input=stn_stack_input)
		
		self.net_depth = net_depth
		self.img_depth= img_depth
		self.z_dim = z_dim

		self.stn_output_depth = self.input_depth*self.nbr_stn
		if self.stn_stack_input :
			self.stn_output_depth += self.input_depth

		# 224
		self.cv1 = conv( self.stn_output_depth, 96, 11, batchNorm=False)
		# 108/109 = E( (224-11+2*1)/2 ) + 1
		self.d1 = nn.Dropout2d(p=0.8)
		self.cv2 = conv( 96, 256, 5)
		# 53 / 54
		self.d2 = nn.Dropout2d(p=0.8)
		self.cv3 = conv( 256, 384, 3)
		# 27 / 27
		self.d3 = nn.Dropout2d(p=0.5)
		self.cv4 = conv( 384, 64, 1)
		# 15
		self.d4 = nn.Dropout2d(p=0.5)
		self.fc = conv( 64, 64, 4, stride=1,pad=0, batchNorm=False)
		# 12
		#self.fc1 = nn.Linear(64 * (12**2), 128)
		#self.fc1 = nn.Linear(64 * (14**2), 128)
		# 120 :
		self.fc1 = nn.Linear(1600, 128)
		# 240 :self.fc1 = nn.Linear(10816, 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, 64)
		self.bn2 = nn.BatchNorm1d(64)
		self.fc3 = nn.Linear(64, self.z_dim)

	def encode(self, x) :
		x = super(STNbasedEncoder,self).forward(x)

		out = F.leaky_relu( self.cv1(x), 0.15)
		out = self.d1(out)
		out = F.leaky_relu( self.cv2(out), 0.15)
		out = self.d2(out)
		out = F.leaky_relu( self.cv3(out), 0.15)
		out = self.d3(out)
		out = F.leaky_relu( self.cv4(out), 0.15)
		out = self.d4(out)
		out = F.leaky_relu( self.fc(out))
		#print(out.size())
		out = out.view( -1, self.num_flat_features(out) )
		#print(out.size())
		out = F.leaky_relu( self.bn1( self.fc1( out) ), 0.15 )
		out = F.leaky_relu( self.bn2( self.fc2( out) ), 0.15)
		out = F.relu(self.fc3( out) )


		return out


	def forward(self,x) :
		return self.encode(x)

	def num_flat_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features


class STNbasedBetaVAE(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(STNbasedBetaVAE,self).__init__()
		self.encoder = STNbasedEncoder(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = Decoder(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
		self.beta = beta
		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
		#veps = Variable( eps, requires_grad=False)
		if self.use_cuda :
			veps = veps.cuda()
		z = mu + veps * torch.exp( log_var/2 )
		return z

	def forward(self,x) :
		h = self.encoder( x)
		mu, log_var = torch.chunk(h, 2, dim=1 )
		z = self.reparameterize( mu,log_var)
		out = self.decoder(z)

		return out, mu, log_var

	def encode(self,x) :
		h = self.encoder( x)
		mu, log_var = torch.chunk(h, 2, dim=1 )
		z = self.reparameterize( mu,log_var)
		
		return z,mu, log_var


	def save(self,path) :
		# Encoder :
		enc_wts = self.encoder.state_dict()
		encpath = path + 'Encoder.weights'
		torch.save( enc_wts, encpath )
		print('Encoder saved at : {}'.format(encpath) )

		# Decoder :
		dec_wts = self.decoder.state_dict()
		decpath = path + 'Decoder.weights'
		torch.save( dec_wts, decpath )
		print('Decoder saved at : {}'.format(decpath) )


	def load(self,path) :
		# Encoder :
		encpath = path + 'Encoder.weights'
		self.encoder.load_state_dict( torch.load( encpath ) )
		print('Encoder loaded from : {}'.format(encpath) )
		
		# Decoder :
		encpath = path + 'Decoder.weights'
		self.decoder.load_state_dict( torch.load( decpath ) )
		print('Decoder loaded from : {}'.format(decpath) )
		


class Rescale(object) :
	def __init__(self, output_size) :
		assert( isinstance(output_size, (int, tuple) ) )
		self.output_size = output_size

	def __call__(self, sample) :
		image = sample
		#image = np.array( sample )
		#h,w = image.shape[:2]

		new_h, new_w = self.output_size

		#img = transform.resize(image, (new_h, new_w) )
		img = image.resize( (new_h,new_w) ) 

		sample = np.reshape( img, (1, new_h, new_w) )

		return sample 




class BasicHeads(nn.Module) :
	def __init__(self,memory, input_dim=256, nbr_heads=1, use_cuda=True,is_read=True) :
		super(BasicHeads,self).__init__()

		self.memory = memory
		self.mem_dim = self.memory.mem_dim
		self.nbr_heads = nbr_heads
		self.input_dim = input_dim
		self.use_cuda = use_cuda
		self.batch_dim=self.memory.batch_dim
		self.is_read = is_read 

		self.generate_ctrl2gate()
		self.reset_prev_w()

	def generate_ctrl2gate(self) :
		if self.is_read is None :
			raise NotImplementedError
		
		if self.is_read :
			# Generate k,beta,g,s,gamma : M + 1 + 1 + 3 + 1 = M+6
			self.head_gate_dim = self.memory.mem_dim+6 
		else :
			# Generate k,beta,g,s,gamma, e, a : M + 1 + 1 + 3 + 1 + M + M = 3*M+6
			self.head_gate_dim = 3*self.memory.mem_dim+6 
		
		self.ctrl2head = nn.Linear(self.input_dim, self.nbr_heads * self.head_gate_dim )
		
	def reset_prev_w(self):
		if self.use_cuda :
			self.prev_w = Variable(torch.zeros(self.batch_dim, self.nbr_heads, self.memory.mem_nbr_slots)).cuda()
		else :
			self.prev_w = Variable(torch.zeros(self.batch_dim, self.nbr_heads, self.memory.mem_nbr_slots))
			
	def reset(self) :
		self.reset_prev_w()

	def write(self,ctlr_input) :
		raise NotImplementedError
	def read(self,ctrl_input) :
		raise NotImplementedError

	def forward(self, ctrl_input) :
		self.ctrl_output = self.ctrl2head(ctrl_input)
		self.ctrl_output = self.ctrl_output.view((-1,self.nbr_heads,self.head_gate_dim))

		self._generate_addressing()

		# Addressing :
		self.wc = self.memory.content_addressing( self.k, self.beta)
		self.w = self.memory.location_addressing( self.prev_w, self.wc, self.g, self.s, self.gamma)

		self.prev_w = self.w

		return self.w 




	def _generate_addressing(self) :
		self.k = self.ctrl_output[:,:,0:self.mem_dim]
		self.beta = F.softplus( self.ctrl_output[:,:,self.mem_dim:self.mem_dim+1] )
		self.g = F.sigmoid( self.ctrl_output[:,:,self.mem_dim+1:self.mem_dim+2] )
		self.s = F.softmax( F.softplus( self.ctrl_output[:,:,self.mem_dim+2:self.mem_dim+5] ) )
		self.gamma = 1+F.softplus( self.ctrl_output[:,:,self.mem_dim+5:self.mem_dim+6] )	

		if not(self.is_read) :
			self.erase = self.ctrl_output[:,:,self.mem_dim+6:2*self.mem_dim+6]
			self.add = self.ctrl_output[:,:,2*self.mem_dim+6:3*self.mem_dim+6]


class ReadHeads(BasicHeads) :
	def __init__(self, memory, nbr_heads=1, input_dim=256, use_cuda=True) :
		super(ReadHeads,self).__init__(memory=memory,input_dim=input_dim,nbr_heads=nbr_heads,is_read=True)

		
	def read(self, ctrl_input) :
		
		w = super(ReadHeads,self).forward(ctrl_input)
		r = self.memory.read( w )

		return r 

class WriteHeads(BasicHeads) :
	def __init__(self, memory, nbr_heads=1, input_dim=256, use_cuda=True) :
		super(WriteHeads,self).__init__(memory=memory,input_dim=input_dim,nbr_heads=nbr_heads,is_read=False)

		
	def write(self, ctrl_input) :
		w = super(WriteHeads,self).forward(ctrl_input)
		self.memory.write( w=w, erase=self.erase, add=self.add )


class NTMController(nn.Module) :
	def __init__(self, input_dim=32, 
						hidden_dim=512, 
						output_dim=32, 
						batch_size=8,
						nbr_layers=1, 
						mem_nbr_slots=128, 
						mem_dim= 32, 
						nbr_read_heads=1, 
						nbr_write_heads=1, 
						use_cuda=True) :

		super(NTMController,self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.batch_size = batch_size
		self.nbr_layers = nbr_layers
		self.mem_nbr_slots = mem_nbr_slots
		self.mem_dim = mem_dim
		self.nbr_read_heads = nbr_read_heads
		self.nbr_write_heads = nbr_write_heads
		self.use_cuda = use_cuda

		self.build_controller()
		

		if self.use_cuda :
			self = self.cuda()

	def initialize_controller(self) :
		self.init_controllerStates()
		
	def build_controller(self) :
		# LSTMs Controller :
		# input = ( x_t, y_{t-1}, r0_{t-1}, ..., rN_{t-1}) / rX = X-th vector read from the memory.
		self.LSTMinput_size = (self.input_dim + self.output_dim) + self.mem_dim*self.nbr_read_heads
		# hidden state / output = controller_output_{t}
		self.LSTMhidden_size = self.hidden_dim
		num_layers = self.nbr_layers
		dropout = 0.5

		self.LSTMs = nn.LSTM(input_size=self.LSTMinput_size,
								hidden_size=self.LSTMhidden_size,
								num_layers=num_layers,
								dropout=dropout,
								batch_first=True,
								bidirectional=False)

		# States :
		self.init_controllerStates()

		# External Outputs :
		self.output_fn = []
		# input = (h_t, r0_{t}, ..., rN_{t})
		self.EXTinput_size = self.LSTMhidden_size + self.mem_dim * self.nbr_read_heads
		self.output_fn.append( nn.Linear(self.EXTinput_size, self.output_dim))
		#self.output_fn.append( nn.Tanh())
		self.output_fn.append( nn.Softmax())
		self.output_fn = nn.Sequential( *self.output_fn)

	def init_controllerStates(self) :
		if self.use_cuda :
			self.ControllerStates = (Variable( torch.zeros(self.nbr_layers,self.batch_size,self.LSTMhidden_size) ).cuda(), Variable( torch.zeros(self.nbr_layers,self.batch_size,self.LSTMhidden_size) ).cuda() )
		else :
			self.ControllerStates = (Variable( torch.zeros(self.nbr_layers,self.batch_size,self.LSTMhidden_size) ),Variable( torch.zeros(self.nbr_layers,self.batch_size,self.LSTMhidden_size) ) )
	
		self.LSTMSs_OUTPUTs = list()
		self.LSTMSs_OUTPUTs.append( (0,self.ControllerStates))

	def reset(self) :
		self.init_controllerStates()

	def forward_controller(self,x) :
		# Input : batch x seq_len x input_dim
		self.input = x['input']
		# Previous Desired Output : batch x seq_len x output_dim
		self.prev_desired_output = x['prev_desired_output']
		# Previously read vector from the memory : batch x seq_len x nbr_read_head * mem_dim
		self.prev_read_vec = x['prev_read_vec']

		ctrl_input = torch.cat( [self.input, self.prev_desired_output, self.prev_read_vec], dim=2)
		
		# Controller States :
		#	hidden states h_{t-1} : batch x nbr_layers x hidden_dim 
		#	cell states c_{t-1} : batch x nbr_layers x hidden_dim 
		prev_hc = self.LSTMSs_OUTPUTs[-1][1]

		# Computations :
		self.LSTMSs_OUTPUTs.append( self.LSTMs(ctrl_input, prev_hc) )
		
		return self.LSTMSs_OUTPUTs[-1]

	def forward_external_output_fn(self, ctrl_output, slots_read) :
		ext_fc_inp = torch.cat( [ctrl_output, slots_read], dim=2).squeeze(1)
		self.output_fn_output = self.output_fn(ext_fc_inp)
		
		return self.output_fn_output


class NTMMemory(nn.Module) :
	def __init__(self, mem_nbr_slots, mem_dim, batch_dim, use_cuda=True) :
		super(NTMMemory,self).__init__()

		self.mem_nbr_slots = mem_nbr_slots
		self.mem_dim = mem_dim
		self.use_cuda = use_cuda
		self.batch_dim = batch_dim

		
		self.initialize_memory()

	def initialize_memory(self) :
		if self.use_cuda :
			self.init_mem = Variable(torch.zeros(self.mem_nbr_slots,self.mem_dim)).cuda()
		else :
			self.init_mem = Variable(torch.zeros(self.mem_nbr_slots,self.mem_dim))
		
		#dev = 1.0/np.sqrt(self.mem_dim+self.mem_nbr_slots)
		#nn.init.uniform( self.init_mem, -dev, dev)
		
		#self.reset()

	def reset(self) :
		self.memory = list()

		if self.use_cuda :
			self.memory.append( self.init_mem.clone().repeat(self.batch_dim,1,1).cuda() )
		else :
			self.memory.append( self.init_mem.clone().repeat(self.batch_dim,1,1) )
		
		self.reading_t = list()
		
	def content_addressing(self,k,beta) :
		nbrHeads = k.size()[1]
		eps = 1e-10
		
		memory_bhSMidx = torch.cat([self.memory[-1].unsqueeze(1)]*nbrHeads, dim=1)
		kmat = torch.cat([k.unsqueeze(2)]*self.mem_nbr_slots, dim=2)
		cossim = F.cosine_similarity( kmat, memory_bhSMidx, dim=3).view((self.batch_dim,nbrHeads,-1))
		w = F.softmax( beta * cossim)
		
		return w 

	def location_addressing(self, pw, wc,g,s,gamma) :
		nbrHeads = g.size()[1]
		
		# Interpolation : 
		wg =  g*wc + (1-g)*pw

		# Shift :
		if self.use_cuda :
			ws = Variable(torch.Tensor(self.batch_dim, nbrHeads, self.mem_nbr_slots) ).cuda()
		else : 
			ws = Variable(torch.Tensor(self.batch_dim, nbrHeads, self.mem_nbr_slots) )
			
		for hidx in range(nbrHeads) :
			res = self._conv_shift(wg[:,hidx], s[:,hidx])
			ws[:,hidx] = res
		
		# Sharpening :
		gamma_Sidx = torch.cat([gamma]*self.mem_nbr_slots, dim=2)
		wgam = ws ** gamma_Sidx
		sumw = torch.sum( wgam, dim=2 )
		w = wgam / sumw
		
		return w		

	def _conv_shift(self,wg,s) :
		size = s.size()[1]
		seq_len = wg.size()[1]
		
		c = torch.cat([wg[-size+1:], wg, wg[:size-1]], dim=1).unsqueeze(1)
		s = s.unsqueeze(1)
		res = F.conv1d( c, s)
		
		ret = res[:,:,1:seq_len+1].squeeze(1)
		
		return ret 
	
	def write(self, w, erase, add) :
		'''
		nmemory = torch.zeros(self.memory[-1].size() )
		if self.use_cuda : nmemory = nmemory.cuda()

		for bidx in range(self.batch_dim) :
			for headidx in range(erase.size()[1]) :
				e = torch.ger(w[bidx][headidx], erase[bidx][headidx])
				a = torch.ger(w[bidx][headidx], add[bidx][headidx])
				nmemory[bidx] = (self.memory[-1][bidx]*(1-e)+a ).data

		if self.use_cuda :
			self.memory.append( Variable(nmemory).cuda())
		else :
			self.memory.append( Variable(nmemory) )
		'''
		nmemory =Variable(torch.zeros(self.memory[-1].size() ) )
		if self.use_cuda : nmemory = nmemory.cuda()

		for bidx in range(self.batch_dim) :
			for headidx in range(erase.size()[1]) :
				e = torch.ger(w[bidx][headidx], erase[bidx][headidx])
				a = torch.ger(w[bidx][headidx], add[bidx][headidx])
				nmemory[bidx] = self.memory[-1][bidx]*(1-e)+a 

		self.memory.append(nmemory)

	def read(self, w) :
		nbrHeads = w.size()[1]
		
		memory_bhSMidx = torch.cat([self.memory[-1].unsqueeze(1)]*nbrHeads, dim=1)
		self.reading_t.append( torch.sum( w.unsqueeze(3) * memory_bhSMidx, dim=2) )
		
		return self.reading_t[-1]


class NTM(nn.Module) :
	def __init__(self,input_dim=32, 
						hidden_dim=512, 
						output_dim=32, 
						nbr_layers=1, 
						mem_nbr_slots=128, 
						mem_dim= 32, 
						nbr_read_heads=1, 
						nbr_write_heads=1, 
						batch_size=32,
						use_cuda=True) :

		super(NTM,self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.nbr_layers = nbr_layers
		self.mem_nbr_slots = mem_nbr_slots
		self.mem_dim = mem_dim
		self.nbr_read_heads = nbr_read_heads
		self.nbr_write_heads = nbr_write_heads
		self.batch_size = batch_size
		self.use_cuda = use_cuda

		self.build_memory()
		self.build_controller()
		self.build_heads()


		if self.use_cuda :
			self = self.cuda()

	def build_memory(self) :
		self.memory = NTMMemory(mem_nbr_slots=self.mem_nbr_slots,mem_dim=self.mem_dim,batch_dim=self.batch_size)
		
	def build_controller(self) :
		self.controller = NTMController( input_dim=self.input_dim, 
											hidden_dim=self.hidden_dim, 
											output_dim=self.output_dim, 
											batch_size=self.batch_size,
											nbr_layers=self.nbr_layers, 
											mem_nbr_slots=self.mem_nbr_slots, 
											mem_dim=self.mem_dim, 
											nbr_read_heads=self.nbr_read_heads, 
											nbr_write_heads=self.nbr_write_heads, 
											use_cuda=self.use_cuda)

	def build_heads(self) :
		self.readHeads = ReadHeads(memory=self.memory,
									nbr_heads=self.nbr_read_heads, 
									input_dim=self.hidden_dim, 
									use_cuda=self.use_cuda)
		
		self.writeHeads = WriteHeads(memory=self.memory,
										nbr_heads=self.nbr_write_heads, 
										input_dim=self.hidden_dim, 
										use_cuda=self.use_cuda)	

	def reset_outputs(self) :
		self.read_outputs = list()

		if self.use_cuda :
			self.read_outputs.append( Variable(torch.zeros(self.batch_size,1,self.nbr_read_heads*self.mem_dim)).cuda() )
		else :
			self.read_outputs.append( Variable(torch.zeros(self.batch_size,1,self.nbr_read_heads*self.mem_dim)) )
		
		self.ext_output = list()
		self.controller_output_state = list()

	def forward(self,x) :
		# NTM_input :
		# 'input' : batch_dim x seq_len x self.input_dim
		# 'prev_desired_output' : batch_dim x seq_len x self.output_dim
		# 'prev_read_vec' : batch_dim x seq_len x self.nbr_read_head * self.mem_dim
		x['prev_read_vec'] = self.read_outputs[-1]

		# Controller Outputs :
		# output : batch_dim x hidden_dim
		# state : ( h, c) 
		self.controller_output_state.append( self.controller.forward_controller(x) )

		# Memory Read :
		# TODO : verify dim :
		# batch_dim x nbr_read_heads * mem_dim :
		self.read_outputs.append( self.readHeads.read(self.controller_output_state[-1][0]) )
		
		# Memory Write :
		self.writeHeads.write(self.controller_output_state[-1][0])

		# External Output Function :
		self.ext_output.append( self.controller.forward_external_output_fn(self.controller_output_state[-1][0], self.read_outputs[-1]) )

		return self.ext_output[-1] 

	def reset(self) :
		self.controller.reset()
		self.memory.reset()
		self.readHeads.reset()
		self.writeHeads.reset()
		self.reset_outputs()
		

	def save(self,path) :
		# Controller :
		ctrl_wts = self.controller.state_dict()
		ctrlpath = path + 'Controller.weights'
		torch.save( ctrl_wts, ctrlpath )
		print('Controller saved at : {}'.format(ctrlpath) )

		# ReadHeads :
		read_wts = self.readHeads.state_dict()
		readpath = path + 'ReadHeads.weights'
		torch.save( read_wts, readpath )
		print('ReadHeads saved at : {}'.format(readpath) )
		# WriteHeads :
		write_wts = self.writeHeads.state_dict()
		writepath = path + 'WriteHeads.weights'
		torch.save( write_wts, writepath )
		print('WriteHeads saved at : {}'.format(writepath) )
		

	def load(self,path) :
		# Controller :
		ctrlpath = path + 'Controller.weights'
		self.controller.load_state_dict( torch.load( ctrlpath ) )
		print('Controller loaded from : {}'.format(ctrlpath) )

		# ReadHeads :
		readpath = path + 'ReadHeads.weights'
		self.readHeads.load_state_dict( torch.load( readpath ) )
		print('ReadHeads loaded from : {}'.format(readpath) )
		# WriteHeads :
		writepath = path + 'WriteHeads.weights'
		self.writeHeads.load_state_dict( torch.load( writepath ) )
		print('WriteHeads loaded from : {}'.format(writepath) )
		

class betaVAE_NTM(nn.Module) :
	def __init__(self, latent_dim, NTMhidden_dim=512, NTMoutput_dim=32, NTMnbr_layers=1, NTMmem_nbr_slots=128, NTMmem_dim= 32, 
						NTMnbr_read_heads=1, NTMnbr_write_heads=1, batch_size=32,
						beta=1.0,net_depth=4,img_dim=224, conv_dim=64, use_cuda=True, img_depth=1) :
		super(betaVAE_NTM,self).__init__()

		self.NTM = NTM(input_dim=latent_dim, 
						hidden_dim=NTMhidden_dim, 
						output_dim=NTMoutput_dim, 
						nbr_layers=NTMnbr_layers, 
						mem_nbr_slots=NTMmem_nbr_slots, 
						mem_dim= NTMmem_dim, 
						nbr_read_heads=NTMnbr_read_heads, 
						nbr_write_heads=NTMnbr_write_heads, 
						batch_size=batch_size,
						use_cuda=use_cuda)
		self.betaVAE = betaVAE(beta=beta,
								net_depth=net_depth,
								img_dim=img_dim, 
								z_dim=latent_dim, 
								conv_dim=conv_dim, 
								use_cuda=use_cuda, 
								img_depth=img_depth)

		self.mu = None
		self.log_var = None 
		self.VAE_output = None

		self.NTM_input = None
		self.ext_output = None

		self.lambdaVAE = 1e-5
		self.lambdaNTM = 1.0 
		self.VAE_loss = None
		self.NTM_loss = None 
		self.total_loss = None  

	def reset(self) :
		self.NTM.reset()
		self.NTM_input = list()
		self.ext_output = list()

	def forward(self,x,target) :
		self.VAE_output, self.mu, self.log_var = self.betaVAE.forward(x)
		
		self.NTM_input.append( dict() )
		# Seq_len = 1 :
		self.NTM_input[-1]['input'] = self.betaVAE.z.unsqueeze(1)
		self.NTM_input[-1]['prev_desired_output'] = target.unsqueeze(1)

		self.ext_output.append( self.NTM.forward(self.NTM_input[-1]) )

		return self.ext_output[-1], self.mu, self.log_var, self.VAE_output  

	def compute_losses(self,x,y,target) :
		self.forward(x=x,target=target)

		# VAE loss :
		self.reconst_loss = F.binary_cross_entropy( self.VAE_output, x, size_average=False)
		# expected log likelyhood :
		try :
			self.expected_log_lik = torch.mean( Bernoulli( self.VAE_output ).log_prob( x ) )
		except Exception as e :
			print(e)
			self.expected_log_lik = Variable(torch.ones(1).cuda())
		
		# kl divergence :
		self.kl_divergence = 0.5 * torch.mean( torch.sum( (self.mu**2 + torch.exp(self.log_var) - self.log_var -1), dim=1) )
		# ELBO :
		#self.elbo = self.expected_log_lik - self.betaVAE.beta * self.kl_divergence
		self.VAE_loss = self.reconst_loss + self.betaVAE.beta*self.kl_divergence
				
		# NTM loss :
		self.NTM_loss = F.binary_cross_entropy_with_logits(input=self.ext_output[-1],target=y)

		# Sum :
		self.total_loss = self.lambdaVAE * self.VAE_loss + self. lambdaNTM * self.NTM_loss 

		return self.total_loss, self.VAE_loss, self.NTM_loss



	def resetNTM(self) :
		self.NTM.reset()

	def save(self,path) :
		try :
			self.NTM.save(path)
		except Exception as e :
			print(e)

		try :	
			self.betaVAE.save(path)
		except Exception as e :
			print(e)

	def load(self,path) :
		try :
			self.NTM.load(path)
		except Exception as e :
			print(e)

		try :
			self.betaVAE.load(path)
		except Exception as e :
			print(e)
			

def test() :
	latent_dim = 10 
	NTMhidden_dim=512 
	NTMoutput_dim=32 
	NTMnbr_layers=1
	NTMmem_nbr_slots=128
	NTMmem_dim= 32

	NTMnbr_read_heads=1
	NTMnbr_write_heads=1
	batch_size=32
	
	beta=1.0
	net_depth=4
	img_dim=224
	conv_dim=64
	use_cuda=True
	img_depth=1

	path = './test'

	betaVAENTM = betaVAE_NTM(latent_dim, NTMhidden_dim=NTMhidden_dim, NTMoutput_dim=NTMoutput_dim, NTMnbr_layers=NTMnbr_layers, 
							NTMmem_nbr_slots=NTMmem_nbr_slots, NTMmem_dim= NTMmem_dim, NTMnbr_read_heads=NTMnbr_read_heads, 
							NTMnbr_write_heads=NTMnbr_write_heads, batch_size=batch_size,
							beta=beta,net_depth=net_depth,img_dim=img_dim, conv_dim=conv_dim, use_cuda=use_cuda, img_depth=img_depth)

	print(betaVAENTM)

	betaVAENTM.save(path)

	betaVAENTM.load(path)
	

if __name__ == '__main__' :
	test()