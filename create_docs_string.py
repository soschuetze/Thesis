import json
import os
import re
import pandas as pd

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

def main():
	docs_raw = create_corpus()
	with open(r'docs_raw.txt', 'w') as fp:
		fp.write("%s\n" % docs_raw)

main()