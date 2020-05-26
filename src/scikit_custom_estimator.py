import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from sklearn.utils import Parallel, delayed

import sent2vec
from src import config


def build_sent2vec_language_model (model_bin=config.RESORCES_DIR + '/pretrained-models/torontobooks_unigrams.bin'):
	model = sent2vec.Sent2vecModel()
	model.load_model(model_bin)

	return model


class Sent2VecTransformer(BaseEstimator, TransformerMixin):

	def __init__ (self, model):
		super(Sent2VecTransformer, self).__init__()
		self.model: sent2vec.Sent2vecModel = model

	def fit (self, X, y=None):
		return self

	def transform (self, X, y=None):
		X_copy = list(X).copy()
		embedded_X_copy = self.model.embed_sentences(X_copy)

		return embedded_X_copy


class CustomFeatureUnion(FeatureUnion):

	def __init__ (self,
				  transformer_list, n_jobs=None,
				  transformer_weights=None, verbose=False,
				  use_in_model=False
				  ):
		super(CustomFeatureUnion, self).__init__(
				transformer_list=transformer_list, n_jobs=n_jobs,
				transformer_weights=transformer_weights, verbose=verbose
		)
		self.use_in_model = use_in_model

	def get_result_as_dictionary (self, Xs):
		res = {}
		for transformer, data in zip(self.transformer_list, Xs):
			res[transformer[0]] = data

		return res

	def fit_transform (self, X, y=None, **fit_params):
		"""Fit all transformers, transform the data and concatenate results.

		Parameters
		----------
		X : iterable or array-like, depending on transformers
			Input data to be transformed.

		y : array-like, shape (n_samples, ...), optional
			Targets for supervised learning.

		Returns
		-------
		X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
			hstack of results of transformers. sum_n_components is the
			sum of n_components (output dimension) over transformers.
		"""
		if self.use_in_model:
			return super(CustomFeatureUnion, self).transform(X)

		results = self._parallel_func(X, y, fit_params, _fit_transform_one)
		if not results:
			# All transformers are None
			return np.zeros((X.shape[0], 0))
		Xs, transformers = zip(*results)

		return self.get_result_as_dictionary(Xs)

	def transform (self, X):
		"""Transform X separately by each transformer, concatenate results.

		Parameters
		----------
		X : iterable or array-like, depending on transformers
		    Input data to be transformed.

		Returns
		-------
		X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
		    hstack of results of transformers. sum_n_components is the
		    sum of n_components (output dimension) over transformers.
		"""
		if self.use_in_model:
			return super(CustomFeatureUnion, self).transform(X)

		Xs = Parallel(n_jobs=self.n_jobs)(
				delayed(_transform_one)(trans, X, None, weight)
				for _, trans, weight in self._iter())

		return self.get_result_as_dictionary(Xs)

