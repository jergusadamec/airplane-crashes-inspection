import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from src import config


def get_logger (name, log_to_file=False, file_log_name='log.txt'):
	logger = logging.getLogger(name)
	if not logger.handlers:
		logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		ch.setFormatter(formatter)
		logger.addHandler(ch)

		if log_to_file:
			fh = logging.FileHandler(config.RESORCES_DIR + '/' + file_log_name, mode='w')
			fh.setLevel(logging.INFO)
			fh.setFormatter(formatter)
			logger.addHandler(fh)

	return logger


def plot_2d_data_with_clusters (data, labels, clusters, save=True):
	scatter_x = np.array(data[:, 0])
	scatter_y = np.array(data[:, 1])

	if labels is not None:
		group = np.array(labels.astype(float))
		cdict = {0: 'dimgray', 1: 'red', 2: 'blue', 3: 'green', 4: 'goldenrod', 5: 'peru', 6: 'coral', 7: 'slateblue',
				 8: 'olivedrab', 9: 'tan', 10: 'indigo'}
		cdict_centroid = {0: 'black', 1: 'darkred', 2: 'navy', 3: 'darkgreen', 4: 'darkgoldenrod', 5: 'orangered', 6: 'darkslateblue', 7: 'darkolivegreen'}

		fig, ax = plt.subplots()
		for g in np.unique(group):
			ix = np.where(group == g)
			ax.scatter(scatter_x[ix], scatter_y[ix], facecolors='none', edgecolors=cdict[g])

			if clusters is not None:
				ax.scatter(
						clusters[int(g)][0], clusters[int(g)][1],
						s=1000, label='Cluster ' + str(int(g)),
						facecolors=cdict[g], edgecolors=cdict_centroid[g], linewidth='3', marker='o'
				)

		ax.legend(prop={'size': 17})

	else:
		fig, ax = plt.subplots()
		ax.scatter(scatter_x, scatter_y)

	plt.tick_params('y', labelsize=20)
	plt.tick_params('x', labelsize=20)

	plt.xlabel('First Principal Component', size=20)
	plt.ylabel('Second Principal Component', size=20)
	plt.title('Visualization of clustered data', size=20)
	if save:
		plt.savefig(config.RESORCES_DIR + '/imgs' + '/kmeans.png')
	else:
		plt.show()


def plot_inertia_score (documents, kmeans, save=True):
	sum_of_squared_distances = []
	K = range(2, 9)

	# for k in K:
	# 	print(k)
	# 	kmeans.set_params(n_clusters=k).fit(documents)
	# 	sum_of_squared_distances.append(kmeans.inertia_)

	sum_of_squared_distances = [2669316.629489561, 2650248.1530749626, 2636322.791191566, 2627937.6716833757, 2612819.7548834835,
	 2603604.1791448733, 2599936.4279423063]

	p1 = Point(x=min(K), y=sum_of_squared_distances[0])
	p2 = Point(x=max(K), y=sum_of_squared_distances[-1])

	plt.tick_params('y', labelsize=30)
	plt.tick_params('x', labelsize=30)

	plt.xlabel('Principal Component 1', size=30)
	plt.ylabel('Principal Component 2', size=30)

	plt.plot(list(K), sum_of_squared_distances, 'bx-')
	plt.plot([p1.x, p2.x], [p1.y, p2.y])
	plt.xlabel('k')
	plt.ylabel('Sum of squared distances within clusters')
	plt.title('Elbow Method For Optimal k')
	if save:
		plt.savefig(config.RESORCES_DIR + '/imgs' + '/kmeans_inertia.png')
	else:
		plt.show()


class Point:

	def __init__ (self, x, y):
		self.x = x
		self.y = y

	def distance_to_line (self, p1, p2):
		x_diff = p2.x - p1.x
		y_diff = p2.y - p1.y
		num = abs(y_diff * self.x - x_diff * self.y + p2.x * p1.y - p2.y * p1.x)
		den = math.sqrt(y_diff ** 2 + x_diff ** 2)
		return num / den


def tf_idf_barplot (terms, idf_score):
	top_n_terms = len(terms)
	y_pos = np.arange(top_n_terms)

	plt.bar(y_pos, idf_score[:top_n_terms], align='center', alpha=0.5)
	plt.xticks(y_pos, terms)

	plt.tick_params('y', labelsize=20)
	plt.tick_params('x', labelsize=20)

	plt.ylabel('Reversed idf score', size=20)
	plt.xlabel('Best terms', size=20)
	plt.title('Top 10 terms by idf score.', size=20)

	plt.show()


def plot_tf_idf_stat (vectorizer, top_n_terms, tf_idf_coefficient_for_reverse_bar_plot):
	terms = vectorizer.get_feature_names()
	idf_score = vectorizer.idf_
	idf_score_indices = (-idf_score).argsort()[:len(terms)]

	terms_by_idf_score_indices = list(map(terms.__getitem__, idf_score_indices))[:top_n_terms]
	idf_score_sorted = list(map(idf_score.__getitem__, idf_score_indices))[:top_n_terms]

	tf_idf_barplot(
			terms=terms_by_idf_score_indices,
			idf_score=idf_score_sorted
	)

	idf_score_indices = idf_score.argsort()
	terms_by_idf_score_indices = list(map(terms.__getitem__, idf_score_indices))[:top_n_terms]

	tf_idf_barplot(
			terms=terms_by_idf_score_indices,
			idf_score=list(map(lambda x: tf_idf_coefficient_for_reverse_bar_plot - x, sorted(idf_score)))
	)


def write_file (content, out_filename, path=config.RESORCES_DIR, mode='a'):
	if not os.path.isdir(path):
		os.mkdir(path)
	filename = os.path.join(path, out_filename)
	with open(filename, mode) as f:
		f.write(content)





