import pandas as pd
import nltk
from nltk.corpus import stopwords

# Tokenizing:

news_sample = pd.read_csv('clean_news_sample.csv')
# print(news_sample['content'])

# print(list(set(news_sample['type'])))

content_example = news_sample['content'].iloc[0]
print('Number of tokens in content[0]:', len(content_example))

# print(nltk.word_tokenize(content_example))

tokens = []
contents = list(news_sample['content'])
# print(contents)

for content in contents:
    tokens = list(set(tokens + nltk.word_tokenize(content)))

print('Number of tokens in all contents:', len(tokens))

# Stopword removal:

stopwords = stopwords.words('english')
# print('Stopwords:', stopwords)

tokens_removed_stopwords = [w for w in tokens if not w in stopwords]

print('Number of tokens, without stopwords:', len(tokens_removed_stopwords))

# Ratio:
def reduction_rate(original, transformed):
    return (len(original) - len(transformed)) / len(original)

print('Reduction rate for stop words:', 
      reduction_rate(tokens, tokens_removed_stopwords))

from nltk.stem.porter import *
stemmer = PorterStemmer()

# plurals = ['dying', 'sleeping', 'walking', 'running', 'hello']
# print([stemmer.stem(plural) for plural in plurals])

tokens_stemmed = list(set([stemmer.stem(token) for token in tokens_removed_stopwords]))
print('Number of unique stemmed tokens:', len(tokens_stemmed))

print('Reduction rate for stopwords, and stemming:', 
      reduction_rate(tokens, tokens_stemmed))


