import pandas as pd
import nltk
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['within the system calls for action', 'disruptive', 'awareness',  'encouragement', 'community gatherings', 'opposition',  'pressuring non-political elites', 'other']
#classifier(sequence_to_classify, candidate_labels)

# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

tweets_df = pd.read_csv("tweets_cleaned.csv", index_col = [0])
tweets_df = tweets_df[tweets_df['id'].notna()]
docs = tweets_df['text'].tolist()

predictions = []
classifications_df = pd.DataFrame(columns=['tweet','classification'])
for i in range(20):
    t = docs[i]
    sequence = t
    prediction = classifier(sequence, candidate_labels)['labels'][0]

    row = {'tweet': t, 'classification': prediction}
    classifications_df = classifications_df.append(row, ignore_index = True)

classifications_df.to_csv("bart_classifications.csv", sep=',', encoding='utf-8')    
