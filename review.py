import nltk
nltk.download('vader_lexicon')

import numpy as np
import pandas as pd

df = pd.read_csv('~/Desktop/Projects/NLP/MovieSA/amazonreviews.tsv', sep='\t')
df.dropna(inplace=True)

blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
df.drop(blanks, inplace=True)

df['label'].value_counts()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sID = SentimentIntensityAnalyzer()

df['scores'] = df['review'].apply(lambda review:sID.polarity_scores(review))

df['compound'] = df['scores'].apply(lambda d:d['compound'])

df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(df['label'], df['comp_score']))

print(classification_report(df['label'], df['comp_score']))
print(confusion_matrix(df['label'], df['comp_score']))
