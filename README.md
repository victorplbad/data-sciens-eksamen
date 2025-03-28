## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

---

## Installation

Download ```995,000_rows.csv``` from https://absalon.ku.dk/courses/80486/files/9275000?wrap=1

Download ```liar_dataset.zip``` from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage
The project as a whole has been compiled into a Jupyter notebook containing all the code and comments suitable for the operation.

#### Part 1: Data processing
- Activate codeblock from Part 1 Task 1 to import pandas, nltk, re, stopwords and activate the two functions full_clean() and is_credible()
- Activate codeblock from Part 1 Task 2 to apply the data processing suite to compile new file ```clean_995000_news.csv```
- - Remember to change filepath of the file to the directory for the ```995,000_rows.csv``` file.
- Use our data exploration suite to find informations on the processed data such as:
- - Amount of reliabne and fake news.
  - Frequency of words. Also creating new file ```frequent_words_10k.csv```
  - Amount of URLs, Dates, Numbers and Emails.
  - Draw a plot of the 10000 most frequent words.
- Activate codeblock from Part 1 Task 4 which uses sklearns train_test_split to see how the data will get split up. 

#### Part 2: Simple logistic regression
- Activate codeblock from Part 2 task 1. Logistic regression classifier to train a logistic regression model on ```clean_995000_news.csv```
- - Using ```frequent_words_10k.csv```  as word frequency vector.
- Apply preprocessing pipeline to our scraped reliable data in Part 2 Task 3.
- - This data has been scraped under exercise 2 and we're applying our full_clean() function into the contents.
  - Activate predictions on our scraped news.
  - Count classified articles.
 
#### Part 3: Advanced model
- Activate codeblock in Part 3 to use Neural Network Model.
  
#### Part 4: Evaluation
- Activate both codeblocks in Part 4 Task 1 to run the logistic regression model and Neural Network model on test-data.
- Activate both codeblock in Part 4 Task 2 to run the models on the LIAR dataset.
- - While also cleaning the liar dataset, creating a new file in the src directory.
---



