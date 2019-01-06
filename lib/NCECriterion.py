import torch
from torch import nn

eps = 1e-7


class NCECriterion(nn.Module):
	"""
	Here we use the calculated NCE Average to calculate loss
	"""
	
	def __init__(self, nLem):
		super(NCECriterion, self).__init__()
		self.nLem = nLem
	
	def forward(self, x, targets):
		"""
		:param x: output N*(k+1) vector, N is batch sze for [v_i,..]
		# K+1 represents [exp(v_i^Tf_i/T)/Z,exp(v_1'^Tf_i/T)/Z,exp(v_2'^Tf_i/T)/Z,...,exp(v_K'^Tf_i/T)/Z]
		:param targets: index
		:return:loss?
		"""
		batchSize = x.size(0)
		K = x.size(1) - 1
		Pnt = 1 / float(self.nLem)
		Pns = 1 / float(self.nLem)
		
		# eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
		Pmt = x.select(1, 0)
		# choose the 0th column,which is the positive loss
		Pmt_div = Pmt.add(K * Pnt + eps)
		lnPmt = torch.div(Pmt, Pmt_div)
		
		# eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
		# Here x.narrow(1,1,K) means we choose x[:,[1:K+1]]
		# then we do exp(v_1'^Tf_i/T)/Z+ K*Pns +eps as the denominator
		Pon_div = x.narrow(1, 1, K).add(K * Pns + eps)
		
		Pon = Pon_div.clone().fill_(K * Pns)
		lnPon = torch.div(Pon, Pon_div)
		
		# equation 6 in ref. A
		lnPmt.log_()
		lnPon.log_()
		
		lnPmtsum = lnPmt.sum(0)
		lnPonsum = lnPon.view(-1, 1).sum(0)
		
		loss = - (lnPmtsum + lnPonsum) / batchSize
		# get
		return loss
