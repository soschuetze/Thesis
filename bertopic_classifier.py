from bertopic import BERTopic
import pandas as pd

def main():
    tweets_df = pd.read_csv("tweets_cleaned.csv", index_col = [0])
    cleaned_tweets = tweets_df['text'].values

    topic_model = BERTopic(nr_topics=5)
    topics, probs = topic_model.fit_transform(cleaned_tweets)

    topics_time = topic_model.topics_over_time(cleaned_tweets, tweets_df["created_at"].tolist())

    print(topic_model.get_topic_info())

main()

