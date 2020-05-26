import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config, utils, preprocessing, scikit_custom_estimator


def parse_args ():
	arg_parser = argparse.ArgumentParser(description='Parser')

	arg_parser.add_argument('--dataset', type=str,metavar='dataset',
							default='/Airplane-Crashes-Since-1908-preprocessed.csv', help='')
	arg_parser.add_argument('--n-clusters', type=int, metavar='n_clusters',
							default=5, help='Number of centroid used in clustering algorithm.')
	arg_parser.add_argument('--n-components', type=int, metavar='n_components',
							default=2, help='Number of components used in dimensionality reduction.')
	arg_parser.add_argument('--n-best-terms', type=int, metavar='n_best_terms',
							default=30, help='Number of most important terms per clusters.')
	arg_parser.add_argument('--max-df', type=int, metavar='max_df',
							default=0.90, help='')
	arg_parser.add_argument('--min-df', type=int, metavar='min_df',
							default=0.007, help='')

	return arg_parser.parse_args()


def print_nbest_terms (most_important):
	for i in range(args.n_best_terms):
		term_index = most_important[-(i)]
		print(' %s' % terms[term_index], end='')
		print()
	print()


def get_transformer_by_name (name):
	for transformer in pipeline.named_steps.features.transformer_list:
		for step in transformer[1].named_steps:
			if name in step:
				return transformer[1].named_steps[step]
	raise ValueError('Transformer not found!')


if __name__ == "__main__":
	args = parse_args()
	args.dimensionality_reduction = True
	TOP_N_TERMS = 10
	TF_IDF_COEFFICIENT_FOR_REVERSE_BAR_PLOT = 3

	df = pd.read_csv(config.DATASET_DIR + args.dataset, encoding='utf-8')
	summaries = df['Summary'].dropna()

	tf_idf_features = [
		('tf_idf_preprocess_summaries',
		 FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('tf_idf_vectorizer', TfidfVectorizer(stop_words=None, max_df=args.max_df, min_df=args.min_df)),
		('scaler', StandardScaler(with_mean=False))
	]
	tf_idf_bi_grams_features = [
		('tf_idf_preprocess_summaries',
		 FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('tf_idf_vectorizer',
		 TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_df=args.max_df, min_df=args.min_df)),
		('scaler', StandardScaler(with_mean=False))
	]
	sent2vec_features = [
		('sent2vec_preprocess_summaries',
		 FunctionTransformer(preprocessing.sent2vec_summaries_preprocessing, validate=False)),
		('sent2vec_transformer',
		 scikit_custom_estimator.Sent2VecTransformer(model=scikit_custom_estimator.build_sent2vec_language_model())),
		('scaler', StandardScaler(with_mean=False))
	]

	pipeline_conf = [
		(
			'features', scikit_custom_estimator.CustomFeatureUnion([
				# ('tf_idf_features', Pipeline(tf_idf_features)),
				# ('tf_idf_bi_grams_features', Pipeline(tf_idf_bi_grams_features)),
				('tf_idf_bi_grams_features_dimensionality_reduction', Pipeline([
					*tf_idf_bi_grams_features,
					('svd', TruncatedSVD(n_components=args.n_components))
				])),
				# ('sent2vec_features_dimensionality_reduction', Pipeline([
				# 	*sent2vec_features,
				# 	('svd', TruncatedSVD(n_components=args.n_components))
				# ])),
				# ('tfidf_features_dimensionality_reduction', Pipeline([
				# 	*tfidf_features,
				# 	('svd', TruncatedSVD(n_components=args.n_components))
				# ]))
			], verbose=True, use_in_model=False)
		)
	]
	pipeline = Pipeline(pipeline_conf, verbose=True)
	pipeline_result = pipeline.fit_transform(summaries)

	gmm = GaussianMixture(
			n_components=args.n_clusters,
			covariance_type='full',
			max_iter=1000,
			n_init=20,
			random_state=40
	)
	labels_dict = {}
	for transformer_name in pipeline_result.keys():
		labels = gmm.fit_predict(pipeline_result[transformer_name])
		labels_dict[transformer_name] = labels
		transformed_summaries = pipeline_result[transformer_name]

		centroids = None

		if 'tf_idf' in transformer_name:
			tf_idf_vectorizer = get_transformer_by_name('tf_idf_vectorizer')
			terms = tf_idf_vectorizer.get_feature_names()

			# utils.plot_tf_idf_stat(tf_idf_vectorizer, TOP_N_TERMS, TF_IDF_COEFFICIENT_FOR_REVERSE_BAR_PLOT)

			centroids = np.zeros((args.n_clusters, len(terms)))
			labels_counter = Counter(labels)
			for cluster_number in range(args.n_clusters):
				print("Cluster {} contains {} samples".format(cluster_number, labels_counter[cluster_number]))

				density = multivariate_normal(
						cov=gmm.covariances_[cluster_number],
						mean=gmm.means_[cluster_number]
				).logpdf(transformed_summaries)

				centroids[cluster_number, :] = get_transformer_by_name('svd').inverse_transform(transformed_summaries)[
					np.argmax(density)]

				most_important = centroids[cluster_number, :].argsort()
				print_nbest_terms(most_important)

		if 'dimensionality_reduction' in transformer_name:
			svd = get_transformer_by_name('svd')
			centroids = svd.transform(centroids) if centroids is not None else None
		else:
			svd = TruncatedSVD(n_components=args.n_components)
			transformed_summaries = svd.fit_transform(transformed_summaries)
			centroids = svd.transform(centroids) if centroids is not None else None

		utils.plot_2d_data_with_clusters(
				data=transformed_summaries,
				labels=labels,
				clusters=centroids,
				save=False
		)
