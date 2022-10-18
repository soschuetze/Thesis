import os
import pandas as pd

def json_file(filename):
    with open(filename) as data_file:
        df = pd.read_json(data_file)
    return df

def merge_dataframes():
    tweets_df = pd.DataFrame()
    directory = 'lda_corpus'
    for filename in os.listdir(directory):
        if ".json" in filename:
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                df = json_file(f)
                frames = [tweets_df,df]
                tweets_df = pd.concat(frames)
    return tweets_df

def main():
    tweets = merge_dataframes()
    print(tweets.head())
    tweets.to_csv("tweets.csv", sep=',', encoding='utf-8')

main()