from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from cleantext import clean
import unidecode

import string
from nltk.tokenize import TweetTokenizer

stopwordsList = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

def main():
	tweets_df = pd.read_csv("tweets_cleaned.csv", index_col = [0])

	tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
		stop_words = 'english',
		lowercase = True,
		token_pattern = r'\b[a-zA-Z]{3,}\b',
		max_df = 0.5,
		min_df = 10)

	dtm_tf = tf_vectorizer.fit_transform(tweets_df['text'].values.astype('U'))
	tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())

	#5 topics
	lda_tf = LatentDirichletAllocation(n_components=5, random_state=0)
	lda_tf.fit(dtm_tf)
	five_topics_lda = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)

	#10 topics
	lda_tf = LatentDirichletAllocation(n_components=10, random_state=0)
	lda_tf.fit(dtm_tf)
	ten_topics_lda = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)

	#20 topics
	lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
	lda_tf.fit(dtm_tf)
	twenty_topics_lda = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)

main()