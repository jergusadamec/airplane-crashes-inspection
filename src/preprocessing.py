import re
from argparse import ArgumentError

import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import scikit_custom_estimator

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


LEMMATIZER = nltk.stem.WordNetLemmatizer()


def _remove_punctuation (summary, regex=r'[!"#$%&\'()*+,./:;<=>?@[\]^`{|}~]'):
	return re.sub(regex, '', summary)


def _remove_whitespace(summary):
	return ' '.join(summary.split())


def _to_lowercase(summary):
	return summary.lower()


def _remove_numbers (summary):
	return re.sub(r'\d+', '', summary)


def _tokenize(summary):
	return nltk.tokenize.word_tokenize(summary)


def _remove_stopwords(tokenized_summary):
	stop_words = set(nltk.corpus.stopwords.words("english"))
	filtered_summary = [word for word in tokenized_summary if word not in stop_words]

	return filtered_summary


def _lemmatize_word(tokenized_summary):
	lemmas = [LEMMATIZER.lemmatize(word, pos='v') for word in tokenized_summary]

	return lemmas


def _preprocess (summary):
	summary = _to_lowercase(summary)
	summary = _remove_punctuation(summary)
	summary = _remove_numbers(summary)
	summary = _remove_whitespace(summary)

	return summary


def sent2vec_summaries_preprocessing (summaries):
	for summary in summaries:
		yield _preprocess(summary)


def tf_idf_summaries_preprocessing (summaries):
	for summary in summaries:
		summary = _preprocess(summary)
		tokenized_summary = _tokenize(summary)
		tokenized_summary = _remove_stopwords(tokenized_summary)
		tokenized_summary = _lemmatize_word(tokenized_summary)

		yield ' '.join(tokenized_summary)


