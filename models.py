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
		outd = 1
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
		self.cvs = []
		outd = conv_dim
		ind= img_depth
		k = 4
		dim = img_dim
		pad = 1
		stride = 2
		self.cvs = []
		self.cvs.append( conv( img_depth, conv_dim, 4, batchNorm=False))
		self.cvs.append( nn.LeakyReLU(0.05) )
		dim = (dim-k+2*pad)/stride +1

		for i in range(1,self.net_depth,1) :
			ind = outd
			outd = conv_dim*(2**i)
			self.cvs.append( conv( ind, outd,k,stride=stride,pad=pad) )
			self.cvs.append( nn.LeakyReLU(0.05) )
			dim = (dim-k+2*pad)/stride +1
		self.cvs = nn.Sequential( *self.cvs)

		ind = outd
		outd = 64
		outdim = 1
		indim = dim
		pad = 0
		stride = 1
		#k = int(indim +2*pad -stride*(outdim-1))
		k=4
		
		#self.fc = conv( ind, outd, k, stride=stride,pad=pad, batchNorm=False)
		# net_depth = 5 :
		#self.fc = nn.Linear( 25088, 2048)
		# net_depth = 3 :
		self.fc = nn.Linear( 8192, 2048)
		self.fc1 = nn.Linear( 2048, 1024)
		self.fc1 = nn.Linear( 2048, 1024)
		self.fc2 = nn.Linear( 1024, z_dim)
		
	def encode(self, x) :
		out = self.cvs(x)

		out = out.view( (-1, self.num_features(out) ) )
		#print(out.size() )

		out = F.leaky_relu( self.fc(out), 0.05 )
		out = F.leaky_relu( self.fc1(out), 0.05 )
		out = self.fc2(out)
		
		return out

	def forward(self,x) :
		return self.encode(x)

	def num_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features

class betaVAE(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAE,self).__init__()
		self.encoder = Encoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth,conv_dim=conv_dim, z_dim=2*z_dim)
		self.decoder = Decoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth, conv_dim=conv_dim, z_dim=z_dim)

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


class DecoderXYS3(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(DecoderXYS3,self).__init__()
		
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

class EncoderXYS3(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(EncoderXYS3,self).__init__()
		
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
		self.fc1 = nn.Linear(64 * (14**2), 128)
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



class betaVAEXYS3(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAEXYS3,self).__init__()
		self.encoder = EncoderXYS3(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = DecoderXYS3(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

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

class STNbasedEncoderXYS3(STNbasedNet) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32, nbr_stn=2, stn_stack_input=True ) :
		super(STNbasedEncoderXYS3,self).__init__(input_dim=img_dim, input_depth=img_depth, nbr_stn=nbr_stn, stn_stack_input=stn_stack_input)
		
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
		self.fc1 = nn.Linear(64 * (14**2), 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, 64)
		self.bn2 = nn.BatchNorm1d(64)
		self.fc3 = nn.Linear(64, self.z_dim)

	def encode(self, x) :
		x = super(STNbasedEncoderXYS3,self).forward(x)

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


class STNbasedBetaVAEXYS3(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(STNbasedBetaVAEXYS3,self).__init__()
		self.encoder = STNbasedEncoderXYS3(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = DecoderXYS3(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

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

class VAE(nn.Module) :
	def __init__(self, net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		#Encoder.__init__(self, img_dim=img_dim, conv_dim=conv_dim, z_dim=2*z_dim)
		#Decoder.__init__(self, img_dim=img_dim, conv_dim=conv_dim, z_dim=z_dim)
		super(VAE,self).__init__()
		self.encoder = Encoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth,conv_dim=conv_dim, z_dim=2*z_dim)
		self.decoder = Decoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth, conv_dim=conv_dim, z_dim=z_dim)

		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
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


class ReadHeads(nn.Module) :
	def __init_(self, nbr_heads=1, input_dim, mem_nbr_slots=32, use_cuda=True) :
		super(ReadHeads,self)__init__()

		self.nbr_heads = nbr_heads
		self.input_dim = input_dim
		self.mem_nbr_slots = mem_nbr_slots
		self.use_cuda = use_cuda

	def read(self,x,Mem) :
		w = x['weights']
		nbr_w = len(w)
		
		r = 0
		for i in range(nbr_w) : 
			r += w[i] * Mem[i]

		return r

class WriteHeads(nn.Module) :
	def __init_(self, nbr_heads=1, input_dim=512, mem_nbr_slots=32, use_cuda=True) :
		super(WriteHeads,self)__init__()

		self.nbr_heads = nbr_heads
		self.input_dim = input_dim
		self.mem_nbr_slots = mem_nbr_slots
		self.use_cuda = use_cuda
		
	def write(self,x,Mem) :
		w = x['weights']
		nbr_w = len(w)
		
		r = 0
		for i in range(nbr_w) : 
			r += w[i] * Mem[i]

		return r


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

	def build_controller(self) :
		# LSTMs Controller :
		# input = ( x_t, y_{t-1}, r0_{t-1}, ..., rN_{t-1}) / rX = X-th vector read from the memory.
		self.LSTMinput_size = (self.input_dim + self.output_dim) + self.mem_dim*self.nbr_read_heads
		# hidden state / output = 
		self.LSTMhidden_size = self.hidden_dim * (self.nbr_read_heads+self.nbr_read_heads)
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
		self.output_fn.append( nn.Tanh())
		self.output_fn = nn.Sequential( *self.output_fn)

	def init_controllerStates(self) :
		self.ControllerStates = dict()
		self.ControllerStates['prev_hc'] = Variable( torch.rand(self.batch_size,self.LSTMhidden_size) )

		if self.use_cuda :
			self.ControllerStates['prev_hc'] = self.ControllerStates['prev_hc'].cuda()

	
	def forward_controller(self,x) :
		# Input : batch x seq_len x input_dim
		self.input = x['input']
		# Previous Desired Output : batch x seq_len x output_dim
		self.prev_desired_output = x['prev_desired_output']
		# Previously read vector from the memory :
		self.prev_read_vec = x['prev_read_vec']

		ctrl_input = torch.cat( [self.input, self.prev_desired_output], dim=2)
		
		# Controller States :
		#	hidden states h_{t-1} : batch x nbr_layers x hidden_dim 
		#	cell states c_{t-1} : batch x nbr_layers x hidden_dim 
		prev_hc = self.ControllerStates['prev_hc']


		# Computations :
		self.LSTMs_output, self.ControllerStates['prev_hc'] = self.LSTMs(ctrl_input, prev_hc)

		return self.LSTMs_output, self.ControllerStates['prev_hc']

	def forward_external_output_fn(self, x) :
		#TODO

class NTM(nn.Module) :
	def __init__(self,input_dim=32, 
						hidden_dim=512, 
						output_dim=32, 
						nbr_layers=1, 
						mem_nbr_slots=128, 
						mem_dim= 32, 
						nbr_read_heads=1, 
						nbr_write_heads=1, 
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
		self.use_cuda = use_cuda

		self.build_controller()
		self.build_heads()


		if self.use_cuda :
			self = self.cuda()

	def build_controller(self) :
		self.controller = NTMController( input_dim=self.input_dim, 
											hidden_dim=self.hidden_dim, 
											output_dim=self.output_dim, 
											nbr_layers=self.nbr_layers, 
											mem_nbr_slots=self.mem_nbr_slots, 
											mem_dim=self.mem_dim, 
											nbr_read_heads=self.nbr_read_heads, 
											nbr_write_heads=self.nbr_write_heads, 
											use_cuda=self.use_cuda) :

	def build_heads(self) :
		self.readHeads = ReadHeads(nbr_heads=self.nbr_read_heads, input_dim=self.hidden_dim, mem_nbr_slots=self.mem_nbr_slots,use_cuda=self.use_cuda)
		self.writeHeads = WriteHeads(nbr_heads=self.nbr_write_heads, input_dim=self.hidden_dim, mem_nbr_slots=self.mem_nbr_slots, use_cuda=self.use_cuda)	


	def forward(self,x) :
		# Input : batch x seq_len x input_dim
		self.input = x['input']
		# Previous Desired Output : batch x seq_len x output_dim
		self.prev_desired_output = x['prev_desired_output']
		ctrl_input = torch.cat( [self.input, self.prev_desired_output], dim=2)
		# Memory : batch x nbr_slots x mem_dim
		self.M = x['M']
		# Controller's States :
		#	vector weights w_{t-1} : 'weights' ; nbr_heads x weight_dim=mem_nbr_slots
		#	hidden states h_{t-1} : 'hidden' : nbr_layers x hidden_dim 
		self.ControllerStates = x['ControllerStates']
		prev_w = self.ControllerStates['prev_w']
		prev_hc = self.ControllerStates['prev_hc']


		# Computations :
		self.LSTMs_outputs = self.LSTMs(ctrl_input, prev_hc)

		outputs = dict()
		outputs['output'] = output
		outputs['M'] = self.M
		outputs['ControllerStates'] = self.ControllerStates

		return outputs


	def reset(self) :
		self.controller.init_controllerStates()


if __name__ == '__main__' :
	test()