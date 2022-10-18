import pandas as pd
import demoji
import numpy as np
import re
import os
import cleantext
 
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

def main():
	tweets = pd.read_csv("tweets.csv", index_col = [0])
	tweets_no_emojis = remove_emojis(tweets)
	tweets_no_mentions = replace_mentions(tweets_no_emojis)
	tweets_no_urls = replace_urls(tweets_no_mentions)
	print(tweets_no_urls['text'].head())
	tweets_no_urls.to_csv("tweets_cleaned.csv", sep=',', encoding='utf-8')

main()
