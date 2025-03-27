import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import pandas as pd
import re

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv("995,000_rows.csv", usecols = ["type", "content"])
df = df.dropna()

omit_types = ['unreliable', 'unknown', 'rumor', 
              '2018-02-10 13:43:39.521661']

for omit_type in omit_types:
    df = df[df.type != omit_type]

stopwords = stopwords.words('english')

def full_clean(text: str, stopwords=stopwords):
    text = text.lower()

    text = re.sub(r'\n', ' ', text) # Remove newlines
    text = re.sub(r' +', ' ', text) # Remove multiple spaces

    text = re.sub(r'([a-zA-Z]+) (\d+)[, ]? (\d{4})', '<DATE>', text) # Date substitution
    text = re.sub(r'([.a-zA-Z0-9]+)@([-a-zA-Z0-9]+).([a-zA-Z]+)', '<EMAIL>', text) # E-Mail substitution
    text = re.sub(r'(https?:\/\/)?(www.)?([-.a-zA-Z0-9]+)[.](co.uk|com|org|net)\/?([\%\-\.\?\_=a-zA-Z0-9\/]+)?', '<URL>', text) # URL substitution
    text = re.sub(r'[0-9]+', '<NUM>', text) # Number substitution

    stemmer = PorterStemmer()                                   # Porter Stemmer from nltk
    tokens = nltk.word_tokenize(text)                           # Tokenizing the text
    tokens = [word for word in tokens if word.isalpha()]        # Removing punctuation
    tokens = [word for word in tokens if word not in stopwords] # Removing Stopwords
    tokens = [stemmer.stem(word) for word in tokens]            # Stemming all the words
    return ' '.join(tokens) # Returning a string consisting of each word in the list

df["content"] = df["content"].apply(full_clean)

def is_credible(article_type):
    if article_type in ['fake', 'satire', 'conspiracy', 'bias', 'hate', 'junksci']:
        return int(0)
    
    elif article_type in ['clickbait', 'political', 'reliable']:
        return int(1)
    
    else:
        return int(2)
    
df['type'] = df['type'].apply(is_credible)

# Store Results

# Save Cleaned Data
df.to_csv("cleaned_995000_news.csv", index=False)

# Display Sample Results
print(df[["content"]].head(10))