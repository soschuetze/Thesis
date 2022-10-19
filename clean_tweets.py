import pandas as pd
import demoji
import numpy as np
import re
import os
import cleantext
import string
import nltk
import re


demoji.download_codes()

def remove_emojis(tweets_df):
	tweets_df['text'] = tweets_df['text'].apply(lambda x: demoji.replace_with_desc(str(x), ":"))

	return tweets_df

def replace_mentions(tweets_df):
	tweets_df['text'] = tweets_df['text'].apply(lambda x: re.sub("@([a-zA-Z0-9_]{1,50})", "MENTION",str(x)))

	return tweets_df

def replace_urls(tweets_df):
	tweets_df['text'] = tweets_df['text'].apply(lambda x: cleantext.replace_urls(str(x), replace_with="URL"))

	return tweets_df

def remove_stop_words(tweets_df, stopwords):
	tweets_df['text'] = tweets_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
	
	return tweets_df

def remove_punctuation(tweets_df):
	tweets_df['text'] = tweets_df['text'].str.replace(r'[^\w\s]','')
	
	return tweets_df

def main():
	stopwords = nltk.corpus.stopwords.words('english')
	punctuation = string.punctuation

	tweets = pd.read_csv("tweets.csv", index_col = [0])
	tweets_no_emojis = remove_emojis(tweets)
	tweets_no_mentions = replace_mentions(tweets_no_emojis)
	tweets_no_urls = replace_urls(tweets_no_mentions)
	tweets_no_stop_words = remove_stop_words(tweets_no_urls, stopwords)
	tweets_no_punctuation = remove_punctuation(tweets_no_stop_words)
	print(tweets_no_punctuation['text'].head())

	tweets_no_punctuation.to_csv("tweets_cleaned.csv", sep=',', encoding='utf-8')

main()
