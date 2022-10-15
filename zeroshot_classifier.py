import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from cleantext import clean
import unidecode
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['within the system calls for action', 'disruptive', 'awareness',  'encouragement', 'community gatherings', 'opposition',  'pressuring non-political elites', 'other']
#classifier(sequence_to_classify, candidate_labels)


# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')


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

print(len(cleaned_tweets))
predictions = []
for i in range(10):
	t = cleaned_tweets[i]
	sequence = t
	prediction = classifier(sequence, candidate_labels)['labels'][0]
	print(f'{t} : {prediction}\n')




