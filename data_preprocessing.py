# -*- coding: utf-8 -*-
"""
@author: Giselle
"""

import re
from sklearn.model_selection import train_test_split
import datasets
#import pandas as pd
#from nltk.corpus import stopwords
#from nltk.stem import RSLPStemmer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.base import BaseEstimator, TransformerMixin

# convert review scores to sentiment
def conv_to_sentiment(review_score):
    review_score = int(review_score)
    
    if review_score < 3:
        return 0
    elif review_score == 3:
        return 2
    else:
        return 1
    
# use regular expression to search for line breaks and replace
def regex_linebreaks(df_list, sub = ' '):
    expr = '[\n\r]'
    return [re.sub(expr, sub, r) for r in df_list]

# use regular expression to search for hyperlinks and replace
def regex_hyperlinks(df_list, sub=' link '):
    expr = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(expr, sub, r) for r in df_list]

# Brazil uses "day month year" order
# use regular expression to search for dates of various expected formats and replacce
def regex_dates(df_list, sub=' date '):
    expr = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(expr, sub, r) for r in df_list]

# use regular expression to search for mentions of money and replace
def regex_money(df_list, sub = ' money '):
    expr = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(expr, sub, r) for r in df_list]

# use regular expression to search for mentions of numbers and replace
def regex_numbers(df_list, sub = ' number '):
    expr = '[0-9]+'
    return [re.sub(expr, sub, r) for r in df_list]

# use regular expression to search for special characters and replace
def regex_special_chars(df_list, sub = ' '):
    expr = '\W'
    return [re.sub(expr, sub, r) for r in df_list]

# use regular expression to search for additional whitespace and replace
def regex_whitespace(df_list, sub = ' '):
    expr1 = '\s+'
    expr2 = '[ \t]+$'

    whitespace = [re.sub(expr1, sub, r) for r in df_list]
    ending_whitespace = [re.sub(expr2, '', r) for r in whitespace]
    return ending_whitespace

def split_data(data):
    test_dataset_size = int(0.1*data.shape[0])
    train_dataset, test = train_test_split(data, test_size=test_dataset_size, random_state=808, stratify=data['review_sentiment'])
    train, validation = train_test_split(train_dataset, test_size=test_dataset_size, random_state=808, stratify=train_dataset['review_sentiment'])
    
    return train, validation, test

def create_tokenized_data(tokenizer, split_data):
    train, validation, test = split_data
    # I could create the dataset directly from pandas, but I will save and load from disk so Datasets com cache it
    # on disk. This is specially useful when you have a very large dataset that does not fit in memory, which is not
    # the case, but I will leave here this way as a demonstration. 
    train.to_csv('train_split.csv')
    validation.to_csv('validation_split.csv')
    test.to_csv('test_split.csv')
    
    dataset = datasets.load_dataset('csv', data_files={'train': 'train_split.csv',
                                                       'validation':'validation_split.csv',
                                                       'test': 'test_split.csv'})
    dataset = dataset.map(lambda example: {'unbiased_text': re.sub(r':[\)\(]+', '', example['review_message'])}, batched=False)
    dataset = dataset.map(lambda examples: tokenizer(examples['unbiased_text']), batched=True)
    dataset = dataset.map(lambda example: {'labels': 1 if example['review_sentiment'] == 'Positive' else 0}, batched=False)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    
    return dataset

"""
--------------------------------------------
------ 2. PROCESSAMENTO DE STOPWORDS -------
--------------------------------------------


# [StopWords] Função para remoção das stopwords e transformação de texto em minúsculas
def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    
    Args:
    ----------
    text: list object where the stopwords will be removed [type: list]
    cached_stopwords: stopwords to be applied on the process [type: list, default: stopwords.words('portuguese')]
    

    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]



--------------------------------------------
-------- 3. APLICAÇÃO DE STEMMING ----------
--------------------------------------------


# [Stemming] Função para aplicação de processo de stemming nas palavras
def stemming_process(text, stemmer=RSLPStemmer()):
    
    Args:
    ----------
    text: list object where the stopwords will be removed [type: list]
    stemmer: type of stemmer to be applied [type: class, default: RSLPStemmer()]
    

    return [stemmer.stem(c) for c in text.split()]



--------------------------------------------
--- 4. EXTRAÇÃO DE FEATURES DE UM CORPUS ---
--------------------------------------------


# [Vocabulary] Função para aplicação de um vetorizador para criação de vocabulário
def extract_features_from_corpus(corpus, vectorizer, df=False):
    
    Args
    ------------
    text: text to be transformed into a document-term matrix [type: string]
    vectorizer: engine to be used in the transformation [type: object]
    

    # Extracting features
    corpus_features = vectorizer.fit_transform(corpus).toarray()
    features_names = vectorizer.get_feature_names()

    # Transforming into a dataframe to give interpetability to the process
    df_corpus_features = None
    if df:
        df_corpus_features = pd.DataFrame(corpus_features, columns=features_names)

    return corpus_features, df_corpus_features



--------------------------------------------
------ 5. DATAVIZ EM ANÁLISE DE TEXTO ------
--------------------------------------------


# [Viz] Função para retorno de DataFrame de contagem por ngram
def ngrams_count(corpus, ngram_range, n=-1, cached_stopwords=stopwords.words('portuguese')):
    
    Args
    ----------
    corpus: text to be analysed [type: pd.DataFrame]
    ngram_range: type of n gram to be used on analysis [type: tuple]
    n: top limit of ngrams to be shown [type: int, default: -1]
    

    # Using CountVectorizer to build a bag of words using the given corpus
    vectorizer = CountVectorizer(stop_words=cached_stopwords, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]

    # Returning a DataFrame with the ngrams count
    count_df = pd.DataFrame(total_list, columns=['ngram', 'count'])
    return count_df



--------------------------------------------
-------- 6. PIPELINE DE DATA PREP ----------
--------------------------------------------


# [TEXT PREP] Classe para aplicar uma série de funções RegEx definidas em um dicionário
class ApplyRegex(BaseEstimator, TransformerMixin):

    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying all regex functions in the regex_transformers dictionary
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)

        return X


# [TEXT PREP] Classe para aplicar a remoção de stopwords em um corpus
class StopWordsRemoval(BaseEstimator, TransformerMixin):

    def __init__(self, text_stopwords):
        self.text_stopwords = text_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]


# [TEXT PREP] Classe para aplicar o processo de stemming em um corpus
class StemmingProcess(BaseEstimator, TransformerMixin):

    def __init__(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]


# [TEXT PREP] Classe para extração de features de um corpus (vocabulário / bag of words / TF-IDF)
class TextFeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer, train=True):
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.train:
            return self.vectorizer.fit_transform(X).toarray()
        else:
            return self.vectorizer.transform(X)
"""