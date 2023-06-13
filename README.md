# Thesis

This project explores the research question of which natural language processing techniques and machine learning models best classify the forms of activism contained in #BlackLivesMatter tweets.

## Table of Contents
* [General Info](#general-information)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)


## General Information
The classification occurs in two segments:
1. Binary classification of irrelevant (0) vs. relevant (1) tweets
2. Multilabel classification of within the system (0), disruptive (1), and encouraging tweets (2)

In the final dataset the labels are:

-1 = Unrelated to #BlackLivesMatter

0 = Within the system calls for action (e.g. voting, contacting an elected official, etc.)

1 = Disruptive calls for action (e.g. protesting, boycotting, etc.) 

2 = Raising awareness/spreading information (e.g. retweet, like, spread the word) 


The project tests traditional machine learning models through the use of preprocessing methods (stop word removal, stemming, and lemmatizint), word embdedings methods (tf-idf and sBert), and various models (perceptrons, knn, svm, and neural networks). It then also tests large language models by fine-tuning Distilbert and looking at BERTopic for unsupervised modeling. 

## Setup
Install the following libraries:

transformers

tensorflow

pandas

numpy

scikit-learn

matplotlib

nltk

sentence-transformers

pytorch

tqdm

DistilBertTokenizer

TFDistilBertForSequenceClassification

## Usage
Download all files and keep in the same folder. The binary_tweets.csv and new_labels_no_zeros.csv are the datasets for binary and multi-label tasks, respectively. These files can be used for both the traditional and LLM tasks.

1. Supervised methods contains the code for running the traditional machine learning methods. Within this, there are options for the different preprocessing techniques and vectorization methods. Comment these out as necessary.
2.  Distilbert_binary_labels.py can be run to retrain the binary Distilbert model.
3.  Distilbert_all_labels.py can be run to retrain the multilabel Distilbert model.
4.  The binary_model folder contains the trained binary Distilbert model.
5.  The multi-label_model folder contains the trained multi-label Distilbert model.
6.  Bertopic_classifier.py and lda_topic_modeling.ipynb can be run for unsupervised classification.
7.  Kmeans_word2vec.py was used to obtain a random sample of 10,000 of the tweets to use in training and testing all models.

##Results
Binary Distilbert: F1-Score of 0.87
Multi-label Distilbert: F1-Score of 0.83




