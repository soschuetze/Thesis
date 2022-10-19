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

tweets_df = pd.read_csv("tweets_cleaned.csv", index_col = [0])
cleaned_tweets = tweets_df['text'].values

predictions = []
with open(r'bart_classifications', 'w') as fp:
    for i in range(20):
        t = cleaned_tweets[i]
        sequence = t
        prediction = classifier(sequence, candidate_labels)['labels'][0]
        tweet_prediction = f"{t} : {prediction}"
        fp.write("%s\n" % tweet_prediction)
