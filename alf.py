import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if needed
nltk.download('punkt')
nltk.download('stopwords')

# Tokenizer function
def tokenizer(df):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = set()

    # Tokenizing each content
    for content in df['content'].dropna():
        tokens.update(nltk.word_tokenize(content.lower()))  # Lowercase for consistency

    print('Total unique tokens:', len(tokens))

    # Stopword removal
    tokens_removed_stopwords = {w for w in tokens if w not in stop_words}
    print('Unique tokens without stopwords:', len(tokens_removed_stopwords))

    # Stemming
    tokens_stemmed = {stemmer.stem(token) for token in tokens_removed_stopwords}
    print('Unique stemmed tokens:', len(tokens_stemmed))

    # Reduction rates
    def reduction_rate(original, transformed):
        return (len(original) - len(transformed)) / len(original) * 100

    print(f"Reduction after stopword removal: {reduction_rate(tokens, tokens_removed_stopwords):.2f}%")
    print(f"Reduction after stopword removal + stemming: {reduction_rate(tokens, tokens_stemmed):.2f}%")
    
    return tokens_stemmed  # Return tokens for further processing if needed

# Read CSV in chunks
filename = '995,000_rows.csv'
chunk_size = 10000

for chunk in pd.read_csv(filename, chunksize=chunk_size):
    tokenizer(chunk)  # Process each chunk
