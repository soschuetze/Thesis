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

def clean_tokenize_text(text, tokenizer, stopwords):
	text = str(text).lower()  # Lowercase words
	text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
	text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
	text = re.sub(r":", "", text)  # Remove multiple spaces in content
	text = re.sub(r"…", "", text)  # Remove ellipsis (and last word)
	text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
	text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
	tweet_tokenizer = TweetTokenizer()
	tokens = tweet_tokenizer.tokenize(text)
	tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
	tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
	tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
	return tokens

def mbkmeans_clusters(X, k, mb, print_silhouette_values):
	km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
	return km, km.labels_

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
	df["tokens"] = df["text_value"].map(lambda x: clean_tokenize_text(x, word_tokenize, custom_stopwords))

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

	clustering, cluster_labels = mbkmeans_clusters(
		X=vectorized_docs,
		k=10,
		mb=1000,
		print_silhouette_values=True
		)

	df_clusters = pd.DataFrame({
    	"text": docs,
    	"tokens": [" ".join(str(text)) for text in token_sample],
    	"cluster": cluster_labels
    	})

	print(df_clusters.sample(20))
	df_clusters.sample(500).to_csv("kmeans_tweets.csv", sep=',', encoding='utf-8')


main()