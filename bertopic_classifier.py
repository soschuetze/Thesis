from bertopic import BERTopic
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from cleantext import clean
import unidecode

def json_file(filename):
    with open(filename) as data_file:
        df = pd.read_json(data_file)
    return df

def create_corpus():
	corpus = []
	directory = 'lda_corpus'
	for file_name in os.listdir(directory):
		if ".json" in file_name:
			f = os.path.join(directory, file_name)
			with open(f) as data_file:
				data = json.load(data_file)
				for i in data:
					corpus.append(i["text"].lower())
	return corpus

docs_raw = create_corpus()

import string
from nltk.tokenize import TweetTokenizer

stopwordsList = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

def cleanTweets(someTweets):
    """Given a string that it's a tweet or many tweets joined together,
    clean it up to use for further analysis.
    """
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(someTweets)
    lowerTokens = [w.lower() for w in tokens]
    stopwordsList = nltk.corpus.stopwords.words('english')
    noStopWords = [w for w in lowerTokens if w not in stopwordsList]
    noSWandPunct = [w for w in noStopWords if w not in string.punctuation]
    
    return noSWandPunct

from nltk.tokenize import TweetTokenizer
cleaned_tweets = []
for tweet in docs_raw:
    cleaned_word_list = cleanTweets(tweet)
    tweet = ' '.join(cleaned_word_list)
    tweet = clean(tweet, no_emoji=True)
    tweet = re.sub(r'http\S+', '', tweet)
    cleaned_tweets.append(tweet)


topic_model = BERTopic(nr_topics=20)

topics, probs = topic_model.fit_transform(cleaned_tweets)

topic_model.visualize_topics()

print(topic_model.get_topic_info())

