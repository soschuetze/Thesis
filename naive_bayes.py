import sys
import json
import math
import random
from numpy import argmax
from collections import Counter
from csv import DictReader
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()
tokenizer = nlp.tokenizer

def load_data(fn):
	tweets = []
	with open(fn,'r') as allTweetsFile:
		csv_dict_reader = DictReader(allTweetsFile)
		for oneTweet in csv_dict_reader:
			tweets.append(oneTweet)

	return tweets

def tokenize(s):
	text = tokenizer(s)
	return [t.text for t in text]

def sort_and_filter_tweets(tweet_dict_list, categories):
	tweets_by_category = {}
	for c in categories:
		tweets_by_category[c] = []
	for tweet in tweet_dict_list:
		tweet_category = tweet["final_label"]
		if tweet_category in categories:
			tweets_by_category[tweet_category].append(tweet)
	return tweets_by_category

def compute_priors(category_tweet_dict):
	total_num_tweets = 0
	category_num_dict = {}
	category_prob_dict = {}
	for category in category_tweet_dict:
		category_num_dict[category] = len(category_tweet_dict[category])
		total_num_tweets = total_num_tweets + len(category_tweet_dict[category])
	for category in category_num_dict:
		category_prob_dict[category] = math.log(category_num_dict[category]/total_num_tweets)
	return category_prob_dict

def count_words_in_tweets(tweet_dict_list, field):
	counter = Counter()
	for tweets in tweet_dict_list:
		counter = counter + Counter(tokenize(tweets[field]))
	return counter

def make_category_counts(category_dict, field):
	category_counts = {}
	for category in category_dict:
		tweet_category_list = category_dict[category]
		category_counts[category] = count_words_in_tweets(tweet_category_list, field)
	return category_counts

def make_vocab(category_counts):
	words = {}
	for category in category_counts:
		for word in category_counts[category]:
			if word not in words:
				words[word] = 1
			else:
				words[word] = words[word] + 1
	
	sorted_dict = {}
	sorted_keys = sorted(words, key=words.get, reverse = True)[:50000]
	for w in sorted_keys:
		sorted_dict[w] = words[w]

	return sorted_dict

def compute_likelihoods(vocab, category_counts):
	new_category_counts = {}
	for c in category_counts:
		new_category_counts[c] = {}
		for g in category_counts[c]:
			if g in vocab:
				new_category_counts[c][g] = category_counts[c][g]

	log_likelihood_dict = {}
	for category in new_category_counts:
		total_words = 0
		category_likelihood_dict = {}
		for words in new_category_counts[category]:
			total_words = total_words + new_category_counts[category][words]
		for w in vocab:
			if w in new_category_counts[category]:
				count_w = new_category_counts[category][w]
			else:
				count_w = 0
			log_likelihood_w = math.log((count_w + 1)/(total_words + len(vocab)))
			category_likelihood_dict[w] = log_likelihood_w
		log_likelihood_dict[category] = category_likelihood_dict
	return log_likelihood_dict

def compute_class_scores(single_tweet_dict, priors, likelihoods, vocab, field):
	class_score_dict = {}
	for category in priors:
		category_sum = priors[category]
		for w in tokenize(single_tweet_dict[field]):
			if w in vocab:
				category_sum = category_sum + likelihoods[category][w]
		class_score_dict[category] = category_sum
	return class_score_dict

def classify_tweet(single_tweet_dict, priors, likelihoods, vocab, field):
	class_score_dict = compute_class_scores(single_tweet_dict, priors, likelihoods, vocab, field)
	extended_score_dict = {}
	extended_score_dict["tweet"] = single_tweet_dict["text"]
	extended_score_dict["field"] = field
	extended_score_dict["gold"] = single_tweet_dict["final_label"]
	max_prob = max(class_score_dict, key = class_score_dict.get)
	extended_score_dict["predicted"] = max_prob
	for category in class_score_dict:
		extended_score_dict[category] = class_score_dict[category]
	return extended_score_dict
		
def classify_all_tweets(category_tweet_dict, priors, likelihoods, vocab, field):
	tweet_classifications = []
	for category in category_tweet_dict:
		for tweet in category_tweet_dict[category]:
			score_dict = classify_tweet(tweet, priors, likelihoods, vocab, field)
			tweet_classifications.append(score_dict)
	return tweet_classifications

def train(training_file, category_list, field):
	tweets = load_data(training_file)
	category_dict = sort_and_filter_tweets(tweets, category_list)
	category_counts = make_category_counts(category_dict, field)
	vocab = make_vocab(category_counts)
	return compute_likelihoods(vocab, category_counts)

def test(test_file, learned_priors, learned_likelihoods, category_list, vocab, field):
	tweets = load_data(test_file)
	category_dict = sort_and_filter_tweets(tweets, category_list)
	tweet_classifications = classify_all_tweets(category_dict, learned_priors, learned_likelihoods, vocab, field)
	return tweet_classifications

def by_class_precision(classifications, category):
	correct_category = 0
	guessed_category_incorrectly = 0
	for tweet in classifications:
		if tweet['gold']==category and tweet['predicted'] == category:
			correct_category =  correct_category + 1
		elif tweet['gold']!=category and tweet['predicted'] == category: #book predicted to be genre but is not actually the genre
			guessed_category_incorrectly = guessed_category_incorrectly + 1
	if correct_category + guessed_category_incorrectly == 0:
		return 0
	else:
		return correct_category/(correct_category + guessed_category_incorrectly)

def by_class_recall(classifications, category):
	correct_category = 0
	didnt_guess_category = 0
	for tweet in classifications:
		if tweet['gold']==category and tweet['predicted'] == category:
			correct_category =  correct_category + 1
		elif tweet['gold']==category and tweet['predicted'] != category:
			didnt_guess_category = didnt_guess_category + 1
	
	if correct_category + didnt_guess_category == 0:
		return 0
	else:
		return correct_category/(correct_category + didnt_guess_category)

def macro_average_metric(classifications, category_list, metric):
	total_percent = 0
	for c in category_list:
		if metric == 'by_class_precision':
			total_percent = total_percent + by_class_precision(classifications, c)
		else:
			total_percent = total_percent + by_class_recall(classifications, c)
	return total_percent/len(category_list)

def display_top_category_scores(classifications, category_list, n, learned_priors):
	for c in category_list:
		final_list = []
		print(c)
		newlist = sorted(classifications, key=lambda d: d[c], reverse=True)
		for book in newlist:
			if tweet['predicted']==c and tweet[c]!=learned_priors[c]:
				final_list.append(tweet['text'])
		print(final_list[:n])
		print("\n")

def main():
	train_fn = "ferguson_train.csv"
	test_fn = "ferguson_test.csv"
	categories = ["0","1","2","3","5","6","8"]
	tweets = load_data(train_fn)
	category_dict = sort_and_filter_tweets(tweets, categories)
	category_counts = make_category_counts(category_dict, "text")
	vocab = make_vocab(category_counts)
	#print(len(vocab))
	learned_likelihoods = train(train_fn, categories, "text")

	#2.6 Question 1
	"""
	for g in genres:
		sorted_words = sorted(genre_counts[g], key = genre_counts[g].get, reverse=True)[:5]
		print(g)
		print(sorted_words)
	"""

	#2.6 Question 2
	"""
	for g in genres:
		print(g)
		sorted_likelihoods = sorted(learned_likelihoods[g], key = learned_likelihoods[g].get, reverse=True)
		top_5 = sorted_likelihoods[:5]
		print(top_5[:5])
	"""
	learned_priors = compute_priors(category_dict)
	classifications = test(test_fn, learned_priors, learned_likelihoods, categories, vocab, "text")
	#print(classifications)
	#4.2 Questions
	
	for c in categories:
		print(c)
		print("prior ", learned_priors[c])
		print("precision ",by_class_precision(classifications, c))
		print("recall ",by_class_recall(classifications, c))
		print("\n")
	

	#4.5 Questions
	print("avg precision")
	avg_precision = macro_average_metric(classifications, categories, 'by_class_precision')
	print(avg_precision)
	print("avg recall")
	avg_recall = macro_average_metric(classifications, categories, 'by_class_recall')
	print(avg_recall)

	f_measure = (2*avg_precision*avg_recall)/(avg_precision+avg_recall)
	print("f-measure")
	print(f_measure)

	#4.7 Questions
	#display_top_genre_scores(classifications, genres, 5, learned_priors)

main()