import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

text = "plus one article on google plus (thanks to ali alfoneh for his assistance in compiling) politics nuclear issue supreme leader tells islamic student associations at foreign universities: “conspiracies and machinations of the enemy, including the ’scientific apartheid’ which they try to subject our nation to, only strengthens the unity of our people.” head of iran’s nuclear energy agency: “conditions not right for implementation of the additional protocol.” more reactors coming online? military admiral habib-allah sayyari, chief of the islamic republic of iran navy: “closure of the hormuz strait is not under consideration” during upcoming war games. added that iranian-made submarines are soon to leave docks in southern iran. seyyed-yahya rahim safavi, former irgc head and current advisor to supreme leader: “risk of an attack against iran is minimal… iran’s defense doctrine does not entail nuclear weapons.” society and culture iranian psychologist, davar sheikhavandi: “window shopping brings girls and boys together in socially dangerous environments, such as shopping malls… which are a prelude to decadence.” the islamic republic’s minister of interior informs the public of changes in the pre-islamic nowruz [new year] holidays in iran. public health minister of health: “iranian-produced aids medicine is registered internationally.” background: see <DATE> “iran news round up.” diplomacy returning from baku, speaker of the iranian parliament gholam-ali haddad adel informed of expansion of iranian-azeri cultural exchanges. aftab-e yazd claims iranian pilgrims to iraq who have obtained a visa at the iraqi embassy in tehran are turned back at the iran-iraq border. “the theater of annapolis a fiasco even before convening.” writes mouthpiece of iranian supreme leader, ayatollah ali khamnei, kayhan. ahmadinejad warns states not to attend. senegal supports iran’s nuclear program. ivory coast foreign minister comes to tehran. press jamal rahimiyan, from the university jihad organization, is appointed as chief editor of iranian student news agency (isna). economy and trade photo of the day"
stopwords = stopwords.words('english')

def full_clean(text, stopwords=stopwords):
    stemmer = PorterStemmer()

    tokens = nltk.word_tokenize(text) # Tokenizing the text
    tokens = [word for word in tokens if word.isalpha()] # Removing punctuation
    tokens = [word for word in tokens if word not in stopwords] # Removing Stopwords
    tokens = [stemmer.stem(word) for word in tokens] # Stemming all the words

    return ' '.join(tokens) # Returning a string consisting of each word in the list

print(full_clean(text))

    