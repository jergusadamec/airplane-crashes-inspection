import argparse
from collections import Counter

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config, preprocessing, scikit_custom_estimator, utils

LOGGER = utils.get_logger('kmeans', log_to_file=True, file_log_name='kmeans_log.txt')


def parse_args ():
	arg_parser = argparse.ArgumentParser(description='Parser')

	arg_parser.add_argument('--dataset', type=str, metavar='dataset',
							default='/Airplane-Crashes-Since-1908-preprocessed.csv', help='')
	arg_parser.add_argument('--alg', type=str, metavar='alg',
							default='kmeans', help='')
	arg_parser.add_argument('--n-clusters', type=int, metavar='n_clusters',
							default=4, help='Number of centroid used in clustering algorithm.')
	arg_parser.add_argument('--plot_inertia', type=bool, metavar='plot_inertia',
							default=False, help='')
	arg_parser.add_argument('--n-components', type=int, metavar='n_components',
							default=2, help='Number of components used in dimensionality reduction.')
	arg_parser.add_argument('--n-best-terms', type=int, metavar='n_best_terms',
							default=10, help='Number of most important terms per clusters.')
	arg_parser.add_argument('--max-df', type=int, metavar='max_df',
							default=0.95, help='')
	arg_parser.add_argument('--min-df', type=int, metavar='min_df',
							default=0.007, help='')

	return arg_parser.parse_args()


def print_nbest_terms (most_important):
	for i in range(args.n_best_terms):
		term_index = most_important[-(i)]
		print(' %s' % terms[term_index], sep='_', end='\t')
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
	LOGGER.info('args -- ' + str(args))
	TOP_N_TERMS = 10
	TF_IDF_COEFFICIENT_FOR_REVERSE_BAR_PLOT = 3

	df = pd.read_csv(config.DATASET_DIR + args.dataset, encoding='utf-8')
	summaries = df['Summary'].dropna()

	dtm_features = [
		('dtm_preprocess_summaries', FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('dtm_vectorizer', CountVectorizer(stop_words=None, max_df=args.max_df, min_df=args.min_df, ngram_range=(1, 3))),
		('scaler', StandardScaler(with_mean=False))
	]
	tf_idf_features = [
		('tf_idf_preprocess_summaries', FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('tf_idf_vectorizer', TfidfVectorizer(stop_words=None, max_df=args.max_df, min_df=args.min_df)),
		('scaler', StandardScaler(with_mean=False))
	]
	tf_idf_bi_grams_features = [
		('tf_idf_preprocess_summaries', FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('tf_idf_vectorizer', TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_df=args.max_df, min_df=args.min_df)),
		('scaler', StandardScaler(with_mean=False))
	]
	tf_idf_3grams_features = [
		('tf_idf_preprocess_summaries', FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('tf_idf_vectorizer', TfidfVectorizer(stop_words=None, ngram_range=(1, 3), max_df=args.max_df, min_df=args.min_df)),
		('scaler', StandardScaler(with_mean=False))
	]
	sent2vec_features = [
		('sent2vec_preprocess_summaries', FunctionTransformer(preprocessing.tf_idf_summaries_preprocessing, validate=False)),
		('sent2vec_transformer', scikit_custom_estimator.Sent2VecTransformer(model=scikit_custom_estimator.build_sent2vec_language_model())),
		('scaler', StandardScaler(with_mean=False))
	]

	pipeline_conf = [
		(
			'features', scikit_custom_estimator.CustomFeatureUnion([
				# ('dtm_ngrams_features', Pipeline(dtm_features)),
				# ('tf_idf_features', Pipeline(tf_idf_features)),
				('tf_idf_bi_grams_features', Pipeline(tf_idf_bi_grams_features)),
				# ('tf_idf_bi_grams_features_dimensionality_reduction', Pipeline([
				# 	*tf_idf_bi_grams_features,
				# 	('svd', TruncatedSVD(n_components=args.n_components))
				# ])),
				# ('tf_idf_3grams_features', Pipeline(tf_idf_3grams_features)),
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
	LOGGER.info('pipeline -- ' + str(pipeline))
	kmeans = KMeans(
			n_clusters=args.n_clusters,
			verbose=0,
			n_init=30,
			max_iter=1000,
			random_state=43,
			n_jobs=-1
	)
	LOGGER.info('model -- ' + str(kmeans))
	labels_dict = {}
	for transformer_name in pipeline_result.keys():
		print('******************************************', transformer_name)
		LOGGER.info('transformer_name -- ' + str(transformer_name))
		labels = kmeans.fit_predict(pipeline_result[transformer_name])
		labels_dict[transformer_name] = labels
		transformed_summaries = pipeline_result[transformer_name]

		# if args.plot_inertia:
		# 	utils.plot_inertia_score(transformed_summaries, kmeans, save=False)

		if 'dimensionality_reduction' in transformer_name:
			svd = get_transformer_by_name('svd')
			centroids = svd.inverse_transform(kmeans.cluster_centers_)
		else:
			centroids = kmeans.cluster_centers_

		LOGGER.info('centroids -- ' + str(centroids))

		tran_name = None
		if 'tf_idf' in transformer_name:
			tran_name = 'tf_idf_vectorizer'
		if 'dtm' in transformer_name:
			tran_name = 'dtm_vectorizer'

		if tran_name is not None:
			tf_idf_vectorizer = get_transformer_by_name(tran_name)
			terms = tf_idf_vectorizer.get_feature_names()
			LOGGER.info('vocabulary size -- ' + str(len(tf_idf_vectorizer.vocabulary_)))
			LOGGER.info('terms -- ' + str(terms))
			# utils.plot_tf_idf_stat(tf_idf_vectorizer, TOP_N_TERMS, TF_IDF_COEFFICIENT_FOR_REVERSE_BAR_PLOT)

			labels_counter = Counter(labels)
			for cluster_number in range(args.n_clusters):
				print("Cluster {} contains {} samples".format(cluster_number, labels_counter[cluster_number]))
				most_important = centroids[cluster_number, :].argsort()
				print_nbest_terms(most_important)

		if 'dimensionality_reduction' in transformer_name:
			svd = get_transformer_by_name('svd')
			centroids = svd.transform(centroids)
		else:
			# svd = TruncatedSVD(n_components=args.n_components)
			svd = PCA(n_components=args.n_components)
			transformed_summaries = svd.fit_transform(transformed_summaries.toarray())
			centroids = svd.transform(centroids)

		utils.plot_2d_data_with_clusters(
				data=transformed_summaries,
				labels=labels,
				clusters=centroids,
				save=False
		)
