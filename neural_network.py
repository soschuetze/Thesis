import sys
import csv
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from csv import DictReader
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

EPOCHS = 5
BATCH_SIZE = 24
MAX_LENGTH = 80
LR = 1e-3
CKPT_DIR = "./ckpt"
NUM_CLASSES = 9

class NN(nn.Module):
	def __init__(self,n_features):
		super(NN,self).__init__()

		self.bert_layer = bert
		self.hidden_layer = nn.Linear(768, 384)
		self.relu1 = nn.ReLU()
		self.hidden_layer2 = nn.Linear(384,192)
		self.relu2 = nn.ReLU()
		self.hidden_layer3 = nn.Linear(192,96)
		self.relu3 = nn.ReLU()
		self.hidden_layer4 = nn.Linear(96,48)
		self.relu4 = nn.ReLU()
		self.flatten_layer = nn.Flatten()
		self.linear_layer = nn.Linear(3840,NUM_CLASSES)
		self.out = nn.LogSoftmax(dim=1)

	def forward(self,x):
		bert = self.bert_layer(x).last_hidden_state
		hidden1 = self.hidden_layer(bert)
		relu1 = self.relu1(hidden1)
		hidden2 = self.hidden_layer2(relu1)
		relu2 = self.relu2(hidden2)
		hidden3 = self.hidden_layer3(relu2)
		relu3 = self.relu3(hidden3)
		hidden4 = self.hidden_layer4(relu3)
		relu4 = self.relu4(hidden4)
		flatten = self.flatten_layer(relu4)
		linear = self.linear_layer(flatten)
		output = self.out(linear)
		return output

#####################################################

def make_data(file_name, label_map):
	tokenized_words = []
	labels = []
	with open(file_name, 'r') as csv_file:
		reader = csv.DictReader(csv_file)
		for tweet in reader:
			tokenized_words.append(tweet['text'])
			labels.append(label_map[tweet['final_label']])
		csv_file.close()

	return tokenized_words, labels

def prep_bert_data(data, max_length):
	tweet_list = []
	for tweet in data:
		
		tokenized_tweet = tokenizer(tweet, max_length = max_length, truncation=True, padding = 'max_length')
		tweet_list.append(torch.tensor(tokenized_tweet['input_ids'], dtype=torch.long))

	return tweet_list

#####################################################

def get_predicted_label_from_predictions(predictions):
	predicted_label = predictions.argmax(1).item()
	return predicted_label

def print_performance_by_class(test_labels,test_predictions):
	acc_list = [[],[],[],[],[],[],[],[],[]]
	for i,p in enumerate(test_predictions):
		predicted_label = get_predicted_label_from_predictions(test_predictions[i])
		true_label = test_labels[i]
		acc = 1 if predicted_label == true_label else 0
		acc_list[int(true_label)].append(acc)
	avg_acc = []
	for a in acc_list:
		if len(a)==0:
			avg_acc.append(0)
		else:
			avg_acc.append(sum(a)/len(a))
	#avg_acc = [sum(a)/len(a) for a in acc_list]
	print("Accuracy by Category:")
	for i,a in enumerate(avg_acc):
		print("Category",i,":",a)

def sample_and_print_predictions(feats,data,labels,model):
	num_indexes = len(data)
	indexes = random.sample(range(0, len(data)), 10)
	predictions = predict(feats,model)
	for i in indexes:
		tweet = data[i]
		prediction = predictions[i].argmax(1).item()
		label = labels[i]

		print(f"{tweet} was predicted to be class {prediction} and is class {label}")

#####################################################

def train(dataloader, model,optimizer,epoch, num_tweets):
	weight = torch.tensor([num_tweets[0], num_tweets[1], num_tweets[2], num_tweets[3], num_tweets[4],num_tweets[5], num_tweets[6], num_tweets[7],num_tweets[8]])
	loss_fn = nn.NLLLoss(weight = weight)
	model.train()
	with tqdm(dataloader, unit="batch") as tbatch:
		for X, y in tbatch:
			X, y = X.to(device), y.to(device)
			# Compute prediction error
			pred = model(X)
			loss = loss_fn(pred, y)

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	torch.save({'epoch':epoch,
		'model_state_dict':model.state_dict(),
		'optimizer_state_dict':optimizer.state_dict(),
		'loss':loss,
		},f"{CKPT_DIR}/ckpt_{epoch}.pt")

def predict(data,model):
	predictions = []
	dataloader = DataLoader(data,batch_size=1)
	with torch.no_grad():
		for X in dataloader:
			X = X.to(device)
			pred = model(X)
			predictions.append(pred)
	return predictions

def test(dataloader,model,dataset_name):
	loss_fn = nn.NLLLoss()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"{dataset_name} Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#####################################################


def make_or_restore_model(nfeat):
	# Either restore the latest model, or create a fresh one
	model = NN(nfeat).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
	checkpoints = [CKPT_DIR + "/" + name for name in os.listdir(CKPT_DIR) if name[-1] == 't']
	
	if checkpoints:
		latest_checkpoint = max(checkpoints, key=os.path.getctime)
		
		print("Restoring from", latest_checkpoint)
		ckpt = torch.load(latest_checkpoint)
		model.load_state_dict(ckpt['model_state_dict'])
		optimizer.load_state_dict(ckpt['optimizer_state_dict'])
		epoch = ckpt['epoch']
		return model,optimizer,epoch
	else:
		print("Creating a new model")
		return model,optimizer,0

#####################################################

def main():

	train_f = 'ferguson_train.csv'
	test_f = 'ferguson_test.csv'

	label_map = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8}
	train_data, train_labels = make_data(train_f,label_map)
	test_data, test_labels = make_data(test_f,label_map)
	
	num_0 = len([t for t in train_labels if t==0])
	num_1 = len([t for t in train_labels if t==1])
	num_2 = len([t for t in train_labels if t==2])
	num_3 = len([t for t in train_labels if t==3])
	num_4 = len([t for t in train_labels if t==4])
	num_5 = len([t for t in train_labels if t==5])
	num_6 = len([t for t in train_labels if t==6])
	num_7 = len([t for t in train_labels if t==7])
	num_8 = len([t for t in train_labels if t==8])

	total_num = num_1 + num_2 + num_3 + num_4 + num_5 + num_6 + num_7 + num_8

	zero_value = num_0 / total_num
	one_value = num_1 / total_num
	two_value = num_2 / total_num
	three_value = num_3 / total_num
	four_value = num_4 / total_num
	five_value = num_5 / total_num
	six_value = num_6 / total_num
	seven_value = num_7 / total_num
	eight_value = num_8 / total_num

	num_category_dict = {
		0: 1/zero_value,
		1: 1/one_value,
		2: 1/two_value,
		3: 1/three_value,
		4: 0,
		5: 1/five_value,
		6: 1/six_value,
		7: 0,
		8: 1/eight_value
	}

	train_feats = prep_bert_data(train_data, MAX_LENGTH)
	test_feats = prep_bert_data(test_data, MAX_LENGTH)
	
	train_dataset = list(zip(train_feats,train_labels))
	test_dataset = list(zip(test_feats,test_labels))


	train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
	test_dataloader = DataLoader(test_dataset,batch_size=1)

	#Retrieve model from a checkpoint or make model
	model,optimizer,epoch_start = make_or_restore_model(MAX_LENGTH) 

	for e in range(epoch_start,EPOCHS):
		print("EPOCH",e)
		model.train()
		train(train_dataloader,model,optimizer,e, num_category_dict)
		model.eval()
		test(train_dataloader,model,"TRAIN")
		test(test_dataloader,model,"TEST")
		test_predictions = predict(test_feats,model)
		print_performance_by_class(test_labels,test_predictions)

	sample_and_print_predictions(test_feats,test_data,test_labels,model)

	y_pred = []
	for i in range(len(test_predictions)):
		y_pred.append(test_predictions[i].argmax(1).item())

	print(precision_score(test_labels, y_pred, average='macro'))
	print(recall_score(test_labels, y_pred, average='macro'))
	print(f1_score(test_labels, y_pred, average='macro'))

	
main()