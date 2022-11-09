import pandas as pd
from csv import DictReader
from collections import defaultdict
from collections import Counter

def getTopUsers(usersDct):
	users = {u: len(usersDct[u]) for u in usersDct}
	return sorted(users.items(), key = lambda pair: pair[1], reverse=True)

def main():

	usersDates = defaultdict(Counter)

	with open("user_tweets_data.csv",'r') as allTweetsFile:
		csv_dict_reader = DictReader(allTweetsFile)
		for oneTweet in csv_dict_reader:
			user = oneTweet['user']
			if oneTweet['created_at_y'] is not None:
				date = oneTweet['created_at_y'][:7]
				usersDates[user][date] += 1

	print("user,months,tweets")

	triples = []

	for u, m in getTopUsers(usersDates)[:500]:
		triples.append((u,m,usersDates[u]))

	triples.sort(key=lambda t: (t[1], max(t[2], key=t[2].get)), reverse=True)

	activists = []

	for u,m,t in triples:
		#print(f"{u},{m},{t}")
		activists.append(u)

	users_df = pd.read_csv("user_tweets_data.csv", index_col = [0])
	activists_df = users_df[users_df['user'].isin(activists)]

	activists_df.to_csv("activists.csv", sep=',', encoding='utf-8')

main()