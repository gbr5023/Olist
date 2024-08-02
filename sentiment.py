# -*- coding: utf-8 -*-
"""
@author: gkredila
"""

"""
-------------------------------------------------------------------------------
-- Import Packages --
-------------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

import torch
import nltk
#import torch.nn.functional as tfunc
#import datasets
from tqdm.notebook import tqdm
#from wordcloud import WordCloud
# if the console cannot find this file, please execute the file once
from postgresql_connection import db_connect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix)
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from matplotlib import rc

from sentiment_data_processing import conv_to_sentiment, regex_linebreaks, regex_hyperlinks, \
    regex_dates, regex_money, regex_numbers, regex_special_chars, regex_whitespace, \
        split_data, create_tokenized_data, evaluate_metrics, input_to_torch_device, \
            eval_preds, viz_conf_matrix, viz_wordcloud

# Torch ML libraries
#import transformers
#import torch
#from torch import nn, optim
#from torch.utils.data import Dataset, DataLoader
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)


"""
-------------------------------------------------------------------------------
-- Load Data --
-------------------------------------------------------------------------------
"""

connection = db_connect()

customer = pd.read_sql_table('customer', connection)
geolocation = pd.read_sql_table('geolocation', connection)
geolocation_state = pd.read_sql_table('geolocation_state', connection)
order_item = pd.read_sql_table('order_item', connection)
orders = pd.read_sql_table('orders', connection)
product = pd.read_sql_table('product', connection)
product_category = pd.read_sql_table('product_category', connection)
review = pd.read_sql_table('review', connection)
seller = pd.read_sql_table('seller', connection)


"""
-------------------------------------------------------------------------------
-- Set Seed for Reproducibility--
-------------------------------------------------------------------------------
"""
rand_seed = 808
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)


"""
-------------------------------------------------------------------------------
-- EDA --
-------------------------------------------------------------------------------
"""
review['review_score'] = review['review_score'].astype('int64').astype('category')

sns.set_theme(style='whitegrid', font_scale=1.2, font='sans-serif')
sns.set_palette(sns.color_palette("Set2"))

fig_scores = sns.countplot(data = review, x="review_score", palette="Set2")
plt.xlabel("Review Score")
plt.title("Review Counts by Score")
plt.show(fig_scores)

review['review_sentiment'] = review.review_score.apply(conv_to_sentiment)

sentiment_classes = ['positive', 'negative']
fig_sentiment = sns.countplot(data = review, x="review_sentiment", palette="Set2").set_xticklabels(sentiment_classes)
plt.xlabel("Review Sentiment")
plt.title("Review Counts by Sentiment Type")
plt.show(fig_sentiment)

review_new = review[review.review_message != 'Not Defined']
#review_new['review_sentiment'] = review_new.review_score.apply(conv_to_sentiment)

# visualize word cloud of positive and negative reviews
viz_wordcloud(review_new.loc[review_new['review_sentiment'] == 'Positive'], "review_message")
viz_wordcloud(review_new.loc[review_new['review_sentiment'] == 'Negative'], "review_message")


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Data Preprocessing --
-------------------------------------------------------------------------------
"""
review_new['review_message'] = regex_linebreaks(review_new['review_message'])
review_new['review_message'] = regex_hyperlinks(review_new['review_message'])
review_new['review_message'] = regex_dates(review_new['review_message'])
review_new['review_message'] = regex_money(review_new['review_message'])
review_new['review_message'] = regex_numbers(review_new['review_message'])
review_new['review_message'] = regex_special_chars(review_new['review_message'])
review_new['review_message'] = regex_whitespace(review_new['review_message'])

review_new.info()
sentiment_type = review_new.review_sentiment.value_counts()
sentiment_type
"""
review_sentiment
Positive    29811
Negative     9283
Name: count, dtype: int64
"""

bert_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

# Store length of each review 
review_token_len = []

# loop through reviews to get tokenized length
for rvw in review_new.review_message:
    tokens = bert_tokenizer.encode(rvw, max_length=512)
    review_token_len.append(len(tokens))

# rework this code
sns.histplot(review_token_len)
plt.xlim([0, 215]);
plt.xlabel('Review Tokenized Length')

# most reviews < 100 tokens, set max length to 175 to cover all
#max_length = 175


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Split Data and Prepare Torch Dataset --
-------------------------------------------------------------------------------
"""

# split into 80-10-10
train_data, validation_data, test_data = split_data(review_new)
#train_data, test_data = train_test_split(review_new, test_size=0.2, random_state=rand_seed)
#validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=rand_seed)

print('Training Set:    ', train_data.shape[0])
print('Validation Set:  ', validation_data.shape[0])
print('Test Set:        ', test_data.shape[0])
"""
Training Set:     31276
Validation Set:   3909
Test Set:         3909
"""

#tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenized_dict = create_tokenized_data(bert_tokenizer, (train_data, validation_data, test_data))


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Build Data Loader to organize data in batches --
-------------------------------------------------------------------------------
"""
train_data_loader = torch.utils.data.DataLoader(tokenized_dict['train'], 
                                                batch_size=16, 
                                                collate_fn=DataCollatorWithPadding(bert_tokenizer))

validation_data_loader = torch.utils.data.DataLoader(tokenized_dict['validation'], 
                                                batch_size=16, 
                                                collate_fn=DataCollatorWithPadding(bert_tokenizer))

test_data_loader = torch.utils.data.DataLoader(tokenized_dict['test'], 
                                          batch_size=16, 
                                          collate_fn=DataCollatorWithPadding(bert_tokenizer))


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Load Model and Set Optimizer, Schedule --
-------------------------------------------------------------------------------
"""
# load the model
bert_model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.train().to(torch_device)

# set up an optimizer and scheduler for controlling the learning rate
epochs_iter = 2 # tried with 1 earlier
num_warmup_steps = 5000
optimizer = AdamW(bert_model.parameters(), lr=5e-6)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 5000, 
                                            num_training_steps = epochs_iter*len(train_data_loader))


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Train --
-------------------------------------------------------------------------------
"""
tracker_epoch = tqdm(range(epochs_iter))
accuracy_loss = 0
signif = 0.95
for i in tracker_epoch:
    tracker_batch = tqdm(enumerate(train_data_loader), 
                         desc=f'Epoch {i}', 
                         total=len(train_data_loader))
    for ind, inputs in tracker_batch:
        inputs = input_to_torch_device(inputs, torch_device)
        optimizer.zero_grad()
        loss, logits = bert_model(**inputs)[:2]
        
        loss.backward()
        optimizer.step()
        
        # calculate a simplified ewma to the loss
        if i == 0 and ind == 0:
            accuracy_loss = loss.cpu().item()
        else:
            accuracy_loss = accuracy_loss * signif + (1-signif) * loss.cpu().item()
        
        tracker_batch.set_postfix(loss=accuracy_loss)
        
        if ind%5000 == 0:
            predictions, labels = eval_preds(bert_model, validation_data_loader, torch_device)
            train_metrics = evaluate_metrics(predictions, labels)
            print(train_metrics)
            

        scheduler.step()
    os.makedirs('C:/Users/18082/OneDrive/Documents/Olist/Epoch/Epoch'+str(i))
    bert_model.save_pretrained('/Epoch'+str(i))  
    

"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Test --
-------------------------------------------------------------------------------
"""
test_predictions, test_labels = eval_preds(bert_model, test_data_loader, torch_device)
test_metrics = evaluate_metrics(test_predictions, test_labels)
print(test_metrics)

transpose_sentiments = ['negative', 'positive']
print(classification_report(test_labels, test_predictions, target_names=transpose_sentiments))
"""
              precision    recall  f1-score   support

    negative       0.74      0.87      0.80       928
    positive       0.96      0.91      0.93      2981

    accuracy                           0.90      3909
   macro avg       0.85      0.89      0.87      3909
weighted avg       0.91      0.90      0.90      3909

VS 2 EPOCHS - Slight improvement

              precision    recall  f1-score   support

    negative       0.79      0.83      0.81       928
    positive       0.95      0.93      0.94      2981

    accuracy                           0.91      3909
   macro avg       0.87      0.88      0.87      3909
weighted avg       0.91      0.91      0.91      3909
"""

conf_mat = confusion_matrix(test_labels, test_predictions)
conf_mat_df = pd.DataFrame(conf_mat, index = transpose_sentiments, columns = transpose_sentiments)
viz_conf_matrix(conf_mat_df)


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Compare Models --
-------------------------------------------------------------------------------
"""

# Set up TfidfVectorizer
def stem_words(doc):
    analyzer = TfidfVectorizer().build_analyzer()
    return (nltk.stem.snowball.PortugueseStemmer().stem(word) for word in analyzer(doc) if word[0]!='@')

tfidf_vectorizer = TfidfVectorizer(
    stop_words=nltk.corpus.stopwords.words('portuguese'), 
    analyzer=stem_words,
    min_df=0.0001, 
    max_features=100000, 
    max_df=0.8)

# create training, validation, and test datasets
xtrain_comp_data = tfidf_vectorizer.fit_transform(train_data['review_message'].apply(lambda s: re.sub(r':[\)\(]+', '', s)))
xvalid_comp_data = tfidf_vectorizer.transform(validation_data['review_message'].apply(lambda s: re.sub(r':[\)\(]+', '', s)))
xtest_comp_data = tfidf_vectorizer.transform(test_data['review_message'].apply(lambda s: re.sub(r':[\)\(]+', '', s)))

ytrain_comp_data = (train_data['review_sentiment'] == 'Positive').astype(int).values
yvalid_comp_data = (validation_data['review_sentiment'] == 'Positive').astype(int).values
ytest_comp_data = (test_data['review_sentiment'] == 'Positive').astype(int).values

log_reg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500, verbose=True)
log_reg.fit(xtrain_comp_data, ytrain_comp_data)
ypredictions_lr = log_reg.predict(xvalid_comp_data)

print(classification_report(yvalid_comp_data, ypredictions_lr, target_names=transpose_sentiments))
"""
              precision    recall  f1-score   support

    negative       0.69      0.92      0.79       928
    positive       0.97      0.87      0.92      2981

    accuracy                           0.88      3909
   macro avg       0.83      0.89      0.85      3909
weighted avg       0.90      0.88      0.89      3909
"""

conf_mat_lr = confusion_matrix(yvalid_comp_data, ypredictions_lr)
conf_mat_df_lr = pd.DataFrame(conf_mat_lr, index = transpose_sentiments, columns = transpose_sentiments)
viz_conf_matrix(conf_mat_df_lr)

# Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(xtrain_comp_data, ytrain_comp_data)
ypredictions_nb = naive_bayes.predict(xvalid_comp_data)
print(classification_report(yvalid_comp_data, ypredictions_nb, target_names=transpose_sentiments))
"""
              precision    recall  f1-score   support

    negative       0.74      0.76      0.75       928
    positive       0.92      0.92      0.92      2981

    accuracy                           0.88      3909
   macro avg       0.83      0.84      0.84      3909
weighted avg       0.88      0.88      0.88      3909
"""

conf_mat_nb = confusion_matrix(yvalid_comp_data, ypredictions_nb)
conf_mat_df_nb = pd.DataFrame(conf_mat_nb, index = transpose_sentiments, columns = transpose_sentiments)
viz_conf_matrix(conf_mat_df_nb)