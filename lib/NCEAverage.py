import math

import torch
from torch import nn
from torch.autograd import Function

from .alias_multinomial import AliasMethod


class NCEFunction(Function):
	@staticmethod
	def forward(self, x, y, memory, idx, params):
		"""
		:param self:
		:param x: feature
		:param y: the label for feature( here is index)
		:param memory: store the (i-1) time's feature
		:param idx: noise index
		:param params: [K, T, -1, momentum]
		:return:
		"""
		K = int(params[0].item())
		T = params[1].item()
		Z = params[2].item()
		
		momentum = params[3].item()
		batchSize = x.size(0)
		outputSize = memory.size(0)
		inputSize = memory.size(1)
		
		# sample positives & negatives
		idx.select(1, 0).copy_(y.data)
		# the 0th column of index is the raw data with index
		# idex is a N*(K+1) ,so the [:0]should be the raw data
		
		# sample correspoinding weights
		weight = torch.index_select(memory, 0, idx.view(-1))
		
		weight.resize_(batchSize, K + 1, inputSize)
		# inner product
		out = torch.bmm(weight, x.data.resize_(batchSize, inputSize, 1))
		out.div_(T).exp_()  # batchSize * self.K+1
		x.data.resize_(batchSize, inputSize)
		
		if Z < 0:
			params[2] = out.mean() * outputSize
			# Here we use monte carlo method to predict the Z, tne normalization problem can be solved by NCE theory
			Z = params[2].item()
			print("normalization constant Z is set to {:.1f}".format(Z))
		
		# out is a N* (K+1) vector, N is batch sze for [v_i,..]
		# K+1 represents [exp(v_i^Tf_i/T)/Z,exp(v_1'^Tf_i/T)/Z,exp(v_2'^Tf_i/T)/Z,...,exp(v_K'^Tf_i/T)/Z]
		
		out.div_(Z).resize_(batchSize, K + 1)
		
		# we save these variables for backward
		self.save_for_backward(x, memory, y, weight, out, params)
		
		
		return out
	
	@staticmethod
	def backward(self, gradOutput):
		"""
		:param self:
		:param gradOutput: Here I just don't know
		:return:
		"""
		x, memory, y, weight, out, params = self.saved_tensors
		K = int(params[0].item())
		T = params[1].item()
		Z = params[2].item()
		momentum = params[3].item()
		batchSize = gradOutput.size(0)
		
		# gradients d Pm / d linear = exp(linear) / Z
		gradOutput.data.mul_(out.data)
		# add temperature
		gradOutput.data.div_(T)
		
		gradOutput.data.resize_(batchSize, 1, K + 1)
		
		# gradient of linear
		gradInput = torch.bmm(gradOutput.data, weight)
		# then we get N*1*128 and resize it to x (N*128)
		gradInput.resize_as_(x)
		
		# update the non-parametric data
		# use momentum 0.5 v_i = 0.5 * v_i^(t-1) + 0.5 * g_i
		weight_pos = weight.select(1, 0).resize_as_(x)
		weight_pos.mul_(momentum)
		weight_pos.add_(torch.mul(x.data, 1 - momentum))
		w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
		# Normalization
		updated_weight = weight_pos.div(w_norm)
		# update memory
		memory.index_copy_(0, y, updated_weight)
		
		return gradInput, None, None, None, None


class NCEAverage(nn.Module):
	
	def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
		# args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m
		super(NCEAverage, self).__init__()
		self.nLem = outputSize
		self.unigrams = torch.ones(self.nLem)
		# noise distribution is a uniform distribution =1/nLem
		self.multinomial = AliasMethod(self.unigrams)
		# Here we give a K and use AliasMethod to do sampling
		self.multinomial.cuda()
		self.K = K
		
		self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
		stdv = 1. / math.sqrt(inputSize / 3)
		self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
	
	def forward(self, x, y):
		"""
		:param x: feature
		:param y: index(label for x)
		:return: out
		"""
		batchSize = x.size(0)
		# use K to select
		idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
		# select batch_size * K random select index
		out = NCEFunction.apply(x, y, self.memory, idx, self.params)
		return out
