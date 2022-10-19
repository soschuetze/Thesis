import os
import random
import re
import string

import nltk
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt


def tokenize_text(text, tokenizer, stopwords):
	tweet_tokenizer = TweetTokenizer()
	tokens = tweet_tokenizer.tokenize(text)
	return tokens

def mbkmeans_clusters(X, k, mb, print_silhouette_values):
	km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
	return km, km.labels_, km.cluster_centers_

def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding"""

    features = []
    for tokens in list_of_docs:
    	zero_vector = np.zeros(model.vector_size)
    	vectors = []
    	for token in tokens:
    		if token in model.wv:
    			try:
    				vectors.append(model.wv[token])
    			except KeyError:
    				continue
    	if vectors:
    		vectors = np.asarray(vectors)
    		avg_vec = vectors.mean(axis=0)
    		features.append(avg_vec)
    	else:
    		features.append(zero_vector)
    return features

def main():
	tweets = pd.read_csv("tweets_cleaned.csv", index_col = [0])
	tweets_text = tweets[['text']].copy()
	custom_stopwords = set(stopwords.words("english"))
	text_columns = ["text"]

	df = tweets_text.copy()
	df["content"] = 0
	df["content"] = df["content"].fillna("")

	for col in text_columns:
		df[col] = df[col].astype(str)

	# Create text column based on title, description, and content
	df["text_value"] = 0
	df["text_value"] = df[text_columns].apply(lambda x: " | ".join(x), axis=1)
	df["tokens"] = df["text_value"].map(lambda x: tokenize_text(x, word_tokenize, custom_stopwords))

	# Remove duplicated after preprocessing
	dash, idx = np.unique(df["tokens"], return_index=True)
	df = df.iloc[idx, :]

	# Remove empty values and keep relevant columns
	df = df.loc[df.tokens.map(lambda x: len(x) > 0), ["text_value", "tokens"]]
	docs = df["text_value"].values
	tokenized_docs = df["tokens"].values

	print(f"Original dataframe: {tweets_text.shape}")
	print(f"Pre-processed dataframe: {df.shape}")

	model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=1)
	token_sample = tokenized_docs
	vectorized_docs = vectorize(token_sample, model=model)
	len(vectorized_docs), len(vectorized_docs[0])

	vector_df = pd.DataFrame(columns=['X','y'])
	for doc in vectorized_docs:
		df2 = {'X': doc[0], 'y': doc[1]}
		vector_df = vector_df.append(df2, ignore_index = True)

	clustering, cluster_labels, centroids = mbkmeans_clusters(
		X=vectorized_docs,
		k=5,
		mb=1000,
		print_silhouette_values=True
		)

	cen_x = [i[0] for i in centroids] 
	cen_y = [i[1] for i in centroids]

	df_clusters = pd.DataFrame({
    	"text": docs,
    	"tokens": [" ".join(str(text)) for text in token_sample],
    	"cluster": cluster_labels
    	})

	df_clusters['cen_x'] = df_clusters.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3], 4:cen_x[4]})
	df_clusters['cen_y'] = df_clusters.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3], 4:cen_y[4]})

	# define and map colors
	colors = ['#DF2020', '#81DF20', '#2095DF', '#DF20DF', '#DC20DF']
	df_clusters['c'] = df_clusters.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4]})

	df_clusters.sample(500).to_csv("kmeans_tweets.csv", sep=',', encoding='utf-8')

	plt.scatter(vector_df.X, vector_df.y, c=df_clusters.c, alpha = 0.6, s=10)
	plt.show()


main()
