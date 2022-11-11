from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np
from csv import DictReader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def load_data(fn):
	DATA = []
	with open(fn,'r') as allTweetsFile:
		csv_dict_reader = DictReader(allTweetsFile)
		for oneTweet in csv_dict_reader:
			DATA.append([oneTweet['text'],oneTweet['final_label']])
		allTweetsFile.close()

	return DATA

def sklearn_knn_predict(trainX, trainy, testX, distance_metric, k):
	knn_model = KNeighborsClassifier(algorithm = 'brute',n_neighbors=k, metric=distance_metric)
	training_model = knn_model.fit(trainX, trainy)
	predicts = training_model.predict(testX)

	return predicts

def get_accuracy(y_true, y_predicted):
    """returns the fraction of correct predictions in y_predicted compared to y_true"""

    total_num = len(y_true)
    num_correct = 0
    for index in range(len(y_true)):
    	if y_true[index]==y_predicted[index]:
    		num_correct = num_correct + 1

    return num_correct/total_num

def knn_grid_search(trainX, trainy, validationX, validationy, distance_metric_list, n_neighbors_list):
    """For each metric in distance_metric_list, and each value k in n_neighbors_list,
    trains knn classifiers with those parameters
    on the training data and computes the accuracy on the validation data.
    Returns a dictionary mapping each value of the hyperparameter pair (metric, k)
    to the accuracy with those hyperparameters on the validation data
    """
    metric_dict = {}
    for m in distance_metric_list:
    	for n in n_neighbors_list:
    		predictions = sklearn_knn_predict(trainX, trainy, validationX, m, n)
    		accuracy = get_accuracy(validationy, predictions)
    		metric_dict[(m, n)] = accuracy
    return metric_dict

def main():
	random.seed(1)
	data = load_data("ferguson_tweets.csv")

	random.shuffle(data)
	corpus = []
	labels = []

	for tweet in data:
		corpus.append(tweet[0])
		labels.append(tweet[1])

	X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.20)


	vectorizer = TfidfVectorizer()
	X_train_vec = vectorizer.fit_transform(X_train).toarray()
	X_test_vec = vectorizer.transform(X_test).toarray()

	#KNN with tuning parameters
	"""
	metric, k = 'None', 0
	validation_accuracy = 0.0

	accuracy_dict = knn_grid_search(X_train_vec, y_train, X_train_vec, y_test, ["euclidean","manhattan"] ,[1,3,5,7,9,11,13,15,17,19])
	minimum_key = min(accuracy_dict, key=accuracy_dict.get)
	min_accuracy = round(accuracy_dict[minimum_key],3)

	print('The best parameters are metric =', minimum_key[0], 'and k =', minimum_key[1], 'with', min_accuracy, 'accuracy on the validation data')
	y_pred = sklearn_knn_predict(X_train_vec, y_train, X_test_vec, minimum_key[0], minimum_key[1])
	"""
	#model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
	#model.fit(X_train_vec, y_train)

	#y_pred = model.predict(X_test_vec)

	#kNN_accuracy = model.score(X_test_vec, y_test)

	#print(precision_score(y_test, y_pred, average='macro'))
	#print(recall_score(y_test, y_pred, average='macro'))
	#print(f1_score(y_test, y_pred, average='macro'))

	#ppn = Perceptron(max_iter=10)
	#ppn.fit(X_train_vec, y_train)

	#clf = MLPClassifier(hidden_layer_sizes=(100,100)).fit(X_train_vec, y_train)
	#y_pred = clf.predict(X_test_vec)

	#y_pred = ppn.predict(X_test_vec)

	print(precision_score(y_test, y_pred, average='macro'))
	print(recall_score(y_test, y_pred, average='macro'))
	print(f1_score(y_test, y_pred, average='macro'))


main()