import numpy as np

def sigmoid(x):
	"""
	A numerically stable version of the logistic sigmoid function.
	"""
	pos_mask = (x >= 0)
	neg_mask = (x < 0)
	z = np.zeros_like(x)
	z[pos_mask] = np.exp(-x[pos_mask])
	z[neg_mask] = np.exp(x[neg_mask])
	top = np.ones_like(x)
	top[neg_mask] = z[neg_mask]
	return top / (1 + z)

def sigmoid_derivative(values): 
	return values*(1-values)

class RNN(object):
	def __init__(self, *args):
		"""
		RNN Object to serialize the NN layers
		Please read this code block and understand how it works
		"""
		self.params = {}
		self.grads = {}
		self.layers = []
		self.paramName2Indices = {}
		self.layer_names = {}

		# process the parameters layer by layer
		layer_cnt = 0
		for layer in args:
			for n, v in layer.params.items():
				if v is None:
					continue
				self.params[n] = v
				self.paramName2Indices[n] = layer_cnt
			for n, v in layer.grads.items():
				self.grads[n] = v
			if layer.name in self.layer_names:
				raise ValueError("Existing name {}!".format(layer.name))
			self.layer_names[layer.name] = True
			self.layers.append(layer)
			layer_cnt += 1
		layer_cnt = 0

	def assign(self, name, val):
		# load the given values to the layer by name
		layer_cnt = self.paramName2Indices[name]
		self.layers[layer_cnt].params[name] = val

	def assign_grads(self, name, val):
		# load the given values to the layer by name
		layer_cnt = self.paramName2Indices[name]
		self.layers[layer_cnt].grads[name] = val

	def get_params(self, name):
		# return the parameters by name
		return self.params[name]

	def get_grads(self, name):
		# return the gradients by name
		return self.grads[name]

	def gather_params(self):
		"""
		Collect the parameters of every submodules
		"""
		for layer in self.layers:
			for n, v in layer.params.iteritems():
				self.params[n] = v

	def gather_grads(self):
		"""
		Collect the gradients of every submodules
		"""
		for layer in self.layers:
			for n, v in layer.grads.iteritems():
				self.grads[n] = v

	def load(self, pretrained):
		""" 
		Load a pretrained model by names 
		"""
		for layer in self.layers:
			if not hasattr(layer, "params"):
				continue
			for n, v in layer.params.iteritems():
				if n in pretrained.keys():
					layer.params[n] = pretrained[n].copy()
					print("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class VanillaRNN(object):
	def __init__(self, input_dim, h_dim, init_scale=0.02, name='vanilla_rnn'):
		"""
		In forward pass, please use self.params for the weights and biases for this layer
		In backward pass, store the computed gradients to self.grads
		name: the name of current layer
		input_dim: input dimension
		h_dim: hidden state dimension
		
		meta: variables needed for the backward pass
		"""
		self.name = name
		self.wx_name = name + "_wx"
		self.wh_name = name + "_wh"
		self.b_name = name + "_b"
		self.input_dim = input_dim
		self.h_dim = h_dim
		self.params = {}
		self.grads = {}
		self.params[self.wx_name] = init_scale * np.random.randn(input_dim, h_dim)
		self.params[self.wh_name] = init_scale * np.random.randn(h_dim, h_dim)
		self.params[self.b_name] = np.zeros(h_dim)
		self.grads[self.wx_name] = None
		self.grads[self.wh_name] = None
		self.grads[self.b_name] = None
		self.meta = None

		
	def step_forward(self, x, prev_h):
		"""
		x: input feature (N, D)
		prev_h: hidden state from the previous timestep (N, H)
		
		next_h: hidden state in the next timestep (N, H)
		meta: variables needed for the backward pass
		"""
		next_h, meta = None, None
		assert np.prod(x.shape[1:]) == self.input_dim, "But got {} and {}".format(
			np.prod(x.shape[1:]), self.input_dim)
		next_h = np.tanh(x.dot(self.params[self.wx_name]) + prev_h.dot(self.params[self.wh_name]) + self.params[self.b_name])
		meta = [x, prev_h, next_h]
		return next_h, meta

	def step_backward(self, dnext_h, meta):
		"""
		dnext_h: gradient w.r.t. next hidden state
		meta: variables needed for the backward pass

		dx: gradients of input feature (N, D)
		dprev_h: gradients of previous hiddel state (N, H)
		dWh: gradients w.r.t. feature-to-hidden weights (D, H)
		dWx: gradients w.r.t. hidden-to-hidden weights (H, H)
		db: gradients w.r.t bias (H,)
		"""

		dx, dprev_h, dWx, dWh, db = None, None, None, None, None
		x, prev_h, next_h = meta
		dtanh = dnext_h*(1.0-next_h**2)
		dx = dtanh.dot(self.params[self.wx_name].T)
		dprev_h = dtanh.dot(self.params[self.wh_name].T)
		dWx = x.T.dot(dtanh)
		dWh = prev_h.T.dot(dtanh)
		db = np.sum(dtanh,axis=0)				
		return dx, dprev_h, dWx, dWh, db

	def forward(self, x, h0):
		"""
		T: number of input sequence 
		D: input sequence dimension
		H: hidden state dimension
		N: batch size 

		x: input feature for the entire timeseries (N, T, D)
		h0: initial hidden state (N, H)
		
		h: hidden states for the entire timeseries (N, T, H)
		self.meta: variables needed for the backward pass
		"""
		h = None
		self.meta = []
		h = np.zeros((x.shape[0],x.shape[1],h0.shape[1]))
		h[:,0,:], meta_i = self.step_forward(x[:,0,:], h0)
		self.meta.append(meta_i)
		for i in range(1, x.shape[1]):
			h[:,i,:], meta_i = self.step_forward(x[:,i,:], h[:,i-1,:])
			self.meta.append(meta_i)
		return h

	def backward(self, dh):
		"""
		dh: gradients of hidden states for the entire timeseries (N, T, H)

		dx: gradient of inputs (N, T, D)
		dh0: gradient w.r.t. initial hidden state (N, H)
		
		self.grads[self.wx_name]: gradient of input-to-hidden weights (D, H)
		self.grads[self.wh_name]: gradient of hidden-to-hidden weights (H, H)
		self.grads[self.b_name]: gradient of biases (H,)
		"""
		dx, dh0, self.grads[self.wx_name], self.grads[self.wh_name], self.grads[self.b_name] = None, None, None, None, None
		N = dh.shape[0]
		T = dh.shape[1]
		H = dh.shape[2]
		D = self.meta[0][0].shape[1]

		dx = np.zeros((N,T,D))
		dh0 = np.zeros((N,H))
		self.grads[self.wx_name] = np.zeros((D,H))
		self.grads[self.wh_name] = np.zeros((H,H))
		self.grads[self.b_name] = np.zeros((H,))
		dnext_h = dh[:,T-1,:]
		for i in reversed(range(0, T)):
			dxi, dhi, dWxi, dWhi, dbi = self.step_backward(dnext_h,self.meta[i])
			dx[:,i,:] = dxi
			if i == 0:
				dh0 = dhi
			else:
				dnext_h = dhi + dh[:,i-1,:]
			self.grads[self.wx_name] += dWxi
			self.grads[self.wh_name] += dWhi
			self.grads[self.b_name] += dbi
		self.meta = []
		return dx, dh0
		
class LSTM(object):
	def __init__(self, input_dim, h_dim, init_scale=0.02, name='lstm'):
		"""
		In forward pass, please use self.params for the weights and biases for this layer
		In backward pass, store the computed gradients to self.grads
		
		name: the name of current layer
		input_dim: input dimension
		h_dim: hidden state dimension
		
		meta: variables needed for the backward pass
		"""
		self.name = name
		self.wx_name = name + "_wx"
		self.wh_name = name + "_wh"
		self.b_name = name + "_b"
		self.input_dim = input_dim
		self.h_dim = h_dim
		self.params = {}
		self.grads = {}
		self.params[self.wx_name] = init_scale * np.random.randn(input_dim, 4*h_dim)
		self.params[self.wh_name] = init_scale * np.random.randn(h_dim, 4*h_dim)
		self.params[self.b_name] = np.zeros(4*h_dim)
		self.grads[self.wx_name] = None
		self.grads[self.wh_name] = None
		self.grads[self.b_name] = None
		self.meta = None

		
	def step_forward(self, x, prev_h, prev_c):
		"""
		x: input feature (N, D)
		prev_h: hidden state from the previous timestep (N, H)
		prev_c: previous cell state (N, H)
		
		self.params[self.wx_name]: input-to-hidden weights (D, 4H)
		self.params[self.wh_name]: hidden-to-hidden weights (H, 4H)
		self.params[self.b_name]: biases (4H,)

		next_h: next hidden state (N, H)
		next_c: next cell state (N, H)

		meta: variables needed for the backward pass
		"""
		next_h, next_c, meta = None, None, None
		#############################################################################
		#Affine Transformation
		A = np.dot(x,self.params[self.wx_name]) + \
				np.dot(prev_h, self.params[self.wh_name]) \
					+ self.params[self.b_name]
		
		H = self.h_dim
		#Input Gate
		a_i = A[:,:H]
		i_t = sigmoid(a_i)
		#Forget Gate
		a_f = A[:,H:2*H]
		f_t = sigmoid(a_f)
		
		#Output/Exposure Gate
		a_o = A[:,2*H:3*H]
		o_t = sigmoid(a_o)
		
		#New Memory
		a_g = A[:,3*H:4*H]
		g_t = np.tanh(a_g)

		#Final Memory 
		c_t = f_t * prev_c + i_t * g_t

		# Activation from the hidden layer
		h_t = o_t * np.tanh(c_t)

		next_h = h_t
		next_c = c_t
		
		meta = [x,prev_h,next_h, prev_c, i_t, f_t, c_t, o_t, g_t]

		return next_h, next_c, meta
	
	def step_backward(self, dnext_h, dnext_c, meta):
		"""
		dnext_h: gradient w.r.t. next hidden state
		meta: variables needed for the backward pass

		dx: gradients of input feature (N, D)
		dprev_h: gradients w.r.t. previous hiddel state (N, H)
		dprev_c: gradients w.r.t. previous cell state (N, H)
		dWh: gradients w.r.t. feature-to-hidden weights (D, 4H)
		dWx: gradients w.r.t. hidden-to-hidden weights (H, 4H)
		db: gradients w.r.t bias (4H,)
		"""
		dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
		#############################################################################

		dsigmoid = np.vectorize(lambda x: x*(1-x))
		dtanh = np.vectorize(lambda x: 1.0-np.tanh(x)**2)
		dtanh2 = np.vectorize(lambda x: 1.0-x**2)
		dh = dnext_h
		x, prev_h, next_h, prev_c, i_t, f_t, c_t, o_t, g_t = meta

		#Derivative of Final memory(c)
		dc_t = dh * o_t * dtanh(c_t) + dnext_c 

		# Derivative of output gate(o)
		do_t = np.tanh(c_t) * dh * dsigmoid(o_t)

		#Derivative of Intermediate memory
		dg_t = dc_t * i_t * dtanh2(g_t)

		#Derivative of Forget Gate
		df_t = dc_t * prev_c * dsigmoid(f_t)
		
		#Derivative of Input Gate
		di_t = dc_t * g_t * dsigmoid(i_t)

		#Derivate w.r.t to bias
		db = np.hstack((np.sum(di_t,axis=0), np.sum(df_t,axis=0), 
						np.sum(do_t,axis=0), np.sum(dg_t,axis=0)))
		
		#Derivate w.r.t Weight-to-Input
		dWx_i = np.dot(x.T,di_t)
		dWx_f = np.dot(x.T,df_t)
		dWx_o = np.dot(x.T,do_t)
		dWx_g = np.dot(x.T,dg_t)
		dWx = np.hstack((dWx_i, dWx_f, dWx_o, dWx_g))

		#Derivate w.r.t Weight-to-hidden-state(Previous)
		dWh_i = np.dot(prev_h.T, di_t)
		dWh_f = np.dot(prev_h.T, df_t)
		dWh_o = np.dot(prev_h.T, do_t)
		dWh_g = np.dot(prev_h.T, dg_t)
		dWh = np.hstack((dWh_i, dWh_f, dWh_o, dWh_g))


		H = self.h_dim
		#Derivative w.r.t Input
		Wx = self.params[self.wx_name]
		dx_i = np.dot(di_t, Wx[:,:H].T)
		dx_f = np.dot(df_t, Wx[:,H:2*H].T)
		dx_o = np.dot(do_t, Wx[:,2*H:3*H].T)
		dx_g = np.dot(dg_t, Wx[:,3*H:4*H].T)
		dx = dx_i + dx_f + dx_o + dx_g

		#Derivitive w.r.t previous hidden state
		Wh = self.params[self.wh_name]
		dprev_h_i = np.dot(di_t, Wh[:,:H].T)
		dprev_h_f = np.dot(df_t, Wh[:,H:2*H].T)
		dprev_h_o = np.dot(do_t, Wh[:,2*H:3*H].T)
		dprev_h_g = np.dot(dg_t, Wh[:,3*H:4*H].T)
		dprev_h = dprev_h_i + dprev_h_f + dprev_h_o + dprev_h_g

		#Derivative w.r.t to previous memory
		dprev_c = dc_t * f_t

		return dx, dprev_h, dprev_c, dWx, dWh, db

	def forward(self, x, h0):
		"""
		Forward pass for an LSTM over an entire sequence of data. 
		T: number of input sequence 
		D: input sequence dimension
		H: hidden state dimension
		N: batch size 

		Please make sure that you define c, which is an internal variable to the LSTM.
        Note that in each timestep, c is passed as input, and the initial state of c is set to zero.

		x: input data (N, T, D)
		h0: initial hidden state (N, H)
		
		self.params[self.wx_name]: weights for input-to-hidden connections (D, 4H)
		self.params[self.wh_name]: weights for hidden-to-hidden connections (H, 4H)
		self.params[self.b_name]: biases (4H,)

		self.meta: variables needed for the backward pass

		h: hidden states for all timesteps of all sequences (N, T, H)
		"""
		h = None
		self.meta = []

		h = np.zeros((x.shape[0],x.shape[1],h0.shape[1]))
		c = np.zeros((x.shape[0],x.shape[1],h0.shape[1]))
		c0 = np.zeros_like(h0)
		h[:,0,:], c[:,0,:], meta_i = self.step_forward(x[:,0,:], h0, c0)
		self.meta.append(meta_i)
		for i in range(1, x.shape[1]):
			h[:,i,:], c[:,i,:], meta_i = self.step_forward(x[:,i,:], h[:,i-1,:], c[:,i-1,:])
			self.meta.append(meta_i)	
		return h

	def backward(self, dh):
		"""
		Backward pass for an LSTM over an entire sequence of data.

		dh: gradients of hidden states for the entire timeseries (N, T, H)
				
		self.meta: variables needed for the backward pass

		dx: gradient of input data (N, T, D)
		dh0: gradient of initial hidden state (N, H)
		
		self.grads[self.wx_name]: gradient w.r.t. input-to-hidden weight (D, 4H)
		self.grads[self.wh_name]: : gradient w.r.t. hidden-to-hidden weight (H, 4H)
		self.grads[self.b_name]: : gradient w.r.t. biases (4H,)
		"""
		dx, dh0 = None, None

		N = dh.shape[0]
		T = dh.shape[1]
		H = dh.shape[2]
		D = self.meta[0][0].shape[1]

		dx = np.zeros((N,T,D))
		dh0 = np.zeros((N,H))


		self.grads[self.wx_name] = np.zeros((D,4*H))
		self.grads[self.wh_name] = np.zeros((H,4*H))
		self.grads[self.b_name] = np.zeros((4*H,))
		dnext_h = dh[:,T-1,:]
		dnext_c = np.zeros_like(dnext_h)

		for i in reversed(range(0, T)):
			dxi, dhi, dnext_c, dWxi, dWhi, dbi = self.step_backward(dnext_h,dnext_c,self.meta[i])
			dx[:,i,:] = dxi
			if i == 0:
				dh0 = dhi
			else:
				dnext_h = dhi + dh[:,i-1,:]
			self.grads[self.wx_name] += dWxi
			self.grads[self.wh_name] += dWhi
			self.grads[self.b_name] += dbi
		self.meta = []
		return dx, dh0
			
		
class word_embedding(object):
	def __init__(self, voc_dim, vec_dim, name="we"):
		"""
		In forward pass, please use self.params for the weights and biases for this layer
		In backward pass, store the computed gradients to self.grads
		
		name: the name of current layer
		voc_dim: words size
		vec_dim: embedding vector dimension
        
		self.meta: variables needed for the backward pass
		"""
		self.name = name
		self.w_name = name + "_w"
		self.voc_dim = voc_dim
		self.vec_dim = vec_dim
		self.params = {}
		self.grads = {}
		self.params[self.w_name] = np.random.randn(voc_dim, vec_dim)
		self.grads[self.w_name] = None
		self.meta = None
		
	def forward(self, x):
		"""
		Forward pass for word embeddings. 
		N: batch size
		T: length of sequences 
		V: number of vocaburary
		D: embedding vector dimension

		x: integer array  (N, T) gives indices of words. Each element idx
		  of x muxt be in the range 0 <= idx < V.
		self.params[self.wx_name]: weight matrix giving word vectors for all words.

		out: array of embedding vectors (N, T, D) giving word vectors for all input words.
		
		self.meta: variables needed for the backward pass
		"""
		out, self.meta = None, None
		out = self.params[self.w_name][x,:]
		self.meta = [out, x]
		return out
		
	def backward(self, dout):
		"""
		Backward pass for word embeddings. Note that we only return gradient for the word embedding
		matrix as we cannot back-propagate into the words.
		
		dout: upstream gradients (N, T, D)
		
		self.meta: variables needed for the backward pass

		self.grads[self.w_name]: gradient of word embedding matrix (V, D).
		"""
		self.grads[self.w_name] = None
		out, x = self.meta 
		self.grads[self.w_name] = np.zeros(self.params[self.w_name].shape)
		np.add.at(self.grads[self.w_name],x,dout)
		
class temporal_fc(object):
	def __init__(self, input_dim, output_dim, init_scale=0.02, name='t_fc'):
		"""
		In forward pass, please use self.params for the weights and biases for this layer
		In backward pass, store the computed gradients to self.grads
		
		name: the name of current layer
		input_dim: input dimension
		output_dim: output dimension
		
		self.meta: variables needed for the backward pass 
		"""
		self.name = name
		self.w_name = name + "_w"
		self.b_name = name + "_b"
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.params = {}
		self.grads = {}
		self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
		self.params[self.b_name] = np.zeros(output_dim)
		self.grads[self.w_name] = None
		self.grads[self.b_name] = None
		self.meta = None
		
	def forward(self, x):
		"""
		Forward pass for a temporal fc layer. 
		N: batch size
		T: number of input sequence 
		D: input vector dimension
		M: output vector dimension

		x: Input data of shape (N, T, D)
		
		self.params[self.w_name]: weights (D, M)
		self.params[self.b_name]: biases (M,)

		out: output data (N, T, M)
		
		self.meta: variables needed for the backward pass 
		"""
		N, T, D = x.shape
		M = self.params[self.b_name].shape[0]
		out = x.reshape(N * T, D).dot(self.params[self.w_name]).reshape(N, T, M) + self.params[self.b_name]
		self.meta = [x, out]
		return out



	def backward(self, dout):
		"""
		Backward pass for temporal fc layer.

		dout: upstream gradients of shape (N, T, M)
		
		self.meta: variables needed for the backward pass 

		dx: gradient of input (N, T, D)
		self.grads[self.w_name]: gradient w.r.t. weights (D, M)
		self.grads[self.b_name]: gradient w.r.t. biases (M,)
		"""
		x, out = self.meta
		N, T, D = x.shape
		M = self.params[self.b_name].shape[0]
		dx = dout.reshape(N * T, M).dot(self.params[self.w_name].T).reshape(N, T, D)
		self.grads[self.w_name] = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
		self.grads[self.b_name] = dout.sum(axis=(0, 1))
		return dx
	
class temporal_softmax_loss(object):
	def __init__(self, dim_average=True):
		"""
		dim_average: if dividing by the input dimension or not
		dLoss: intermediate variables to store the scores
		label: ground truth label for classification task
		"""
		self.dim_average = dim_average  # if average w.r.t. the total number of features
		self.dLoss = None
		self.label = None

	def forward(self, feat, label, mask):
		""" Some comments """
		loss = None
		N, T, V = feat.shape
		feat_flat = feat.reshape(N * T, V)
		label_flat = label.reshape(N * T)
		mask_flat = mask.reshape(N * T)

		probs = np.exp(feat_flat - np.max(feat_flat, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), label_flat]))
		if self.dim_average:
			loss /= N

		self.dLoss = probs.copy()
		self.label = label
		self.mask = mask
		return loss

	def backward(self):
		N, T = self.label.shape
		dLoss = self.dLoss
		if dLoss is None:
			raise ValueError("No forward function called before for this module!")
		dLoss[np.arange(dLoss.shape[0]), self.label.reshape(N * T)] -= 1.0
		if self.dim_average:
			dLoss /= N
		dLoss *= self.mask.reshape(N * T)[:, None]
		self.dLoss = dLoss
		return dLoss.reshape(N, T, -1)



