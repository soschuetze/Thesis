from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import pandas as pd

def main():
    sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    tweets_df = pd.read_csv("tweets_cleaned.csv", index_col = [0])
    tweets_df = tweets_df[tweets_df['id'].notna()]
    docs = tweets_df['text'].tolist()

    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    topic_model = BERTopic(nr_topics=10)
    topics, probs = topic_model.fit_transform(docs, embeddings)


    topic_model_info = topic_model.get_topic_info()
    print(topic_model_info)

    topic_model_info.to_csv("bertopic_classifications.csv", sep=',', encoding='utf-8')

main()


