# -*- coding: utf-8 -*-
"""
@author: Giselle
"""

import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

"""
--------------------------------------------
---------- 1. FUNÇÕES PARA REGEX -----------
--------------------------------------------
"""

# [RegEx] Padrão para encontrar quebra de linha e retorno de carro (\n ou \r)
def re_breakline(text_list, text_sub=' '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    return [re.sub('[\n\r]', text_sub, r) for r in text_list]


# [RegEx] Padrão para encontrar sites ou hiperlinks
def re_hiperlinks(text_list, text_sub=' link '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


# [RegEx] Padrão para encontrar datas nos mais diversos formatos (dd/mm/yyyy, dd/mm/yy, dd.mm.yyyy, dd.mm.yy)
def re_dates(text_list, text_sub=' data '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, text_sub, r) for r in text_list]


# [RegEx] Padrão para encontrar valores financeiros (R$ ou  $)
def re_money(text_list, text_sub=' dinheiro '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


# [RegEx] Padrão para encontrar números
def re_numbers(text_list, text_sub=' numero '):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    return [re.sub('[0-9]+', text_sub, r) for r in text_list]


# [RegEx] Padrão para encontrar a palavra "não" em seus mais diversos formatos
def re_negation(text_list, text_sub=' negação '):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', text_sub, r) for r in text_list]


# [RegEx] Padrão para limpar caracteres especiais
def re_special_chars(text_list, text_sub=' '):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    return [re.sub('\W', text_sub, r) for r in text_list]


# [RegEx] Padrão para limpar espaços adicionais
def re_whitespaces(text_list):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    """

    # Applying regex
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end


"""
--------------------------------------------
------ 2. PROCESSAMENTO DE STOPWORDS -------
--------------------------------------------
"""

# [StopWords] Função para remoção das stopwords e transformação de texto em minúsculas
def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    """
    Args:
    ----------
    text: list object where the stopwords will be removed [type: list]
    cached_stopwords: stopwords to be applied on the process [type: list, default: stopwords.words('portuguese')]
    """

    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]


"""
--------------------------------------------
-------- 3. APLICAÇÃO DE STEMMING ----------
--------------------------------------------
"""

# [Stemming] Função para aplicação de processo de stemming nas palavras
def stemming_process(text, stemmer=RSLPStemmer()):
    """
    Args:
    ----------
    text: list object where the stopwords will be removed [type: list]
    stemmer: type of stemmer to be applied [type: class, default: RSLPStemmer()]
    """

    return [stemmer.stem(c) for c in text.split()]


"""
--------------------------------------------
--- 4. EXTRAÇÃO DE FEATURES DE UM CORPUS ---
--------------------------------------------
"""

# [Vocabulary] Função para aplicação de um vetorizador para criação de vocabulário
def extract_features_from_corpus(corpus, vectorizer, df=False):
    """
    Args
    ------------
    text: text to be transformed into a document-term matrix [type: string]
    vectorizer: engine to be used in the transformation [type: object]
    """

    # Extracting features
    corpus_features = vectorizer.fit_transform(corpus).toarray()
    features_names = vectorizer.get_feature_names()

    # Transforming into a dataframe to give interpetability to the process
    df_corpus_features = None
    if df:
        df_corpus_features = pd.DataFrame(corpus_features, columns=features_names)

    return corpus_features, df_corpus_features


"""
--------------------------------------------
------ 5. DATAVIZ EM ANÁLISE DE TEXTO ------
--------------------------------------------
"""

# [Viz] Função para retorno de DataFrame de contagem por ngram
def ngrams_count(corpus, ngram_range, n=-1, cached_stopwords=stopwords.words('portuguese')):
    """
    Args
    ----------
    corpus: text to be analysed [type: pd.DataFrame]
    ngram_range: type of n gram to be used on analysis [type: tuple]
    n: top limit of ngrams to be shown [type: int, default: -1]
    """

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


"""
--------------------------------------------
-------- 6. PIPELINE DE DATA PREP ----------
--------------------------------------------
"""

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
--------------------------------------------
--- 7. UTILITIES FOR SENTIMENT ANALYSIS ----
--------------------------------------------
"""

# Defining a function to plot the sentiment of a given phrase
def sentiment_analysis(text, pipeline, vectorizer, model):
    """
    Args
    -----------
    text: text string / phrase / review comment to be analysed [type: string]
    pipeline: text prep pipeline built for preparing the corpus [type: sklearn.Pipeline]
    model: classification model trained to recognize positive and negative sentiment [type: model]
    """

    # Applying the pipeline
    if type(text) is not list:
        text = [text]
    text_prep = pipeline.fit_transform(text)
    matrix = vectorizer.transform(text_prep)

    # Predicting sentiment
    pred = model.predict(matrix)
    proba = model.predict_proba(matrix)

    # Plotting the sentiment and its score
    fig, ax = plt.subplots(figsize=(5, 3))
    if pred[0] == 1:
        text = 'Positive'
        class_proba = 100 * round(proba[0][1], 2)
        color = 'seagreen'
    else:
        text = 'Negative'
        class_proba = 100 * round(proba[0][0], 2)
        color = 'crimson'
    ax.text(0.5, 0.5, text, fontsize=50, ha='center', color=color)
    ax.text(0.5, 0.20, str(class_proba) + '%', fontsize=14, ha='center')
    ax.axis('off')
    ax.set_title('Sentiment Analysis', fontsize=14)
    plt.show()