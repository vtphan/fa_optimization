#------------------------------------------------------------------------------
#
# Author: Vinhthuy Phan, 2021
#
#------------------------------------------------------------------------------
from .buckets import *
import joblib

class Solution:
	def __init__(self, min_merit, max_merit, min_need, max_need, min_diff):
		self.merit_buckets = None
		self.need_buckets = None
		self.min_merit = min_merit
		self.max_merit = max_merit
		self.min_need = min_need
		self.max_need = max_need
		self.min_diff = min_diff

    #--------------------------------------------------------------------------
	def set_need_buckets(self, b):
		self.need_buckets = Buckets(b, self.min_need, self.max_need, self.min_diff)

    #--------------------------------------------------------------------------
	def set_merit_buckets(self, b):
		self.merit_buckets = Buckets(b, self.min_merit, self.max_merit, self.min_diff)

    #--------------------------------------------------------------------------
	def set_merit_buckets_to_quantile_means(self, model):
		n = len(model.merit_quantiles) - 1
		tmp = model.X[['amount_merit']].groupby(model.stats.merit_cat).mean()
		b = [0]*n
		for i in range(1, n):
			b[i] = round(tmp.amount_merit[i],0)
		self.merit_buckets = Buckets(b, self.min_merit, self.max_merit, self.min_diff)

    #--------------------------------------------------------------------------

	def copy(self):
		s = Solution(self.min_merit, self.max_merit, self.min_need, self.max_need, self.min_diff)
		if self.merit_buckets is not None:
			s.merit_buckets = self.merit_buckets.copy()
		if self.need_buckets is not None:
			s.need_buckets = self.need_buckets.copy()
		return s 

	#--------------------------------------------------------------------------

	def save(self, filename):
		joblib.dump(self, filename)

    #--------------------------------------------------------------------------
	@classmethod
	def load(cls, filename):
		return joblib.load(filename)

    #--------------------------------------------------------------------------
	def __repr__(self):
		return 'Need: {}\nMerit: {}\n'.format(self.need_buckets, self.merit_buckets)


        