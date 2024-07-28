# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 01:05:47 2024

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
import nltk
import torch
import torch.nn.functional as tfunc
import datasets
from tqdm.notebook import tqdm

from postgresql_connection import db_connect
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from matplotlib import rc

from data_preprocessing import conv_to_sentiment, regex_linebreaks, regex_hyperlinks, \
    regex_dates, regex_money, regex_numbers, regex_special_chars, regex_whitespace, \
        split_data, create_tokenized_data

# Torch ML libraries
import transformers
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
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

sentiment_classes = ['negative', 'neutral', 'positive']
fig_sentiment = sns.countplot(data = review, x="review_sentiment", palette="Set2").set_xticklabels(sentiment_classes)
plt.xlabel("Review Sentiment")
plt.title("Review Counts by Sentiment Type")
plt.show(fig_sentiment)


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Data Preprocessing --
-------------------------------------------------------------------------------
"""
review_new = review[review.review_message != 'Not Defined']
#review_new['review_sentiment'] = review_new.review_score.apply(conv_to_sentiment)

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

bert_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

#tokenized_data = create_tokenized_data(bert_tokenizer, (train_data, validation_data, test_data))

# Store length of each review 
review_token_len = []

# loop through reviews to get tokenized length
for rvw in review_new.review_message:
    tokens = bert_tokenizer.encode(rvw, max_length=512)
    review_token_len.append(len(tokens))

sns.distplot(review_token_len)
plt.xlim([0, 215]);
plt.xlabel('Review Tokenized Length')

# most reviews < 100 tokens, set max length to 175 to cover all
max_length = 175


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Prepare Torch Dataset --
-------------------------------------------------------------------------------
"""
class OlistData(Dataset):
    # Constructor Function 
    def __init__(self, review_msgs, sentiments, bert_tokenizer, max_length):
        self.review_msgs = review_msgs
        self.sentiments = sentiments
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
    
    # Length magic method
    def __len__(self):
        return len(self.review_msgs)
    
    # get item magic method
    def __getitem__(self, i):
        msg = str(self.review_msgs[i])
        sentiment = self.sentiments[i]
        
        # Encoded format to be returned 
        encoding = self.bert_tokenizer.encode_plus(
            msg,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_msg': msg,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiments': torch.tensor(sentiment, dtype=torch.long)
        }
    
    
"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Split Data --
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
-------------------------------------------------------------------------------
-- Sentiment Analysis - Build Data Loader to organize data in batches --
-------------------------------------------------------------------------------
"""
def build_loader(data, bert_tokenizer, max_length, batch_size):
    olist = OlistData(
        review_msgs = data.review_message.to_numpy(),
        sentiments = data.review_sentiment.to_numpy(),
        bert_tokenizer = bert_tokenizer,
        max_length = max_length
    )
    
    return DataLoader(
        olist,
        batch_size = batch_size,
        num_workers = 0
    )

# Create data loaders
train_data_loader = build_loader(train_data, bert_tokenizer, max_length, batch_size=16)
val_data_loader = build_loader(validation_data, bert_tokenizer, max_length, batch_size=16)
test_data_loader = build_loader(test_data, bert_tokenizer, max_length, batch_size=16)

bert_model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased')

# iterations
epochs_iter = 10
total_steps = len(train_data_loader) * epochs_iter

# AdamW is an optimizing algorithm that corrects weight decay
optimizer = AdamW(bert_model.parameters(), lr=5e-6)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model.train().to(torch_device)


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Eval Func --

NEED TO UPDATE THIS CODE
-------------------------------------------------------------------------------
"""
def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def send_inputs_to_device(inputs, torch_device):
    return {key:tensor.to(torch_device) for key, tensor in inputs.items()}

def predict(bert_model, val_data_loader, torch_device):
    with torch.no_grad():
        bert_model.eval()
        preds = []
        labels = []
        validation_losses = []
        for inputs in val_data_loader:
            labels.append(inputs['labels'].numpy())
            
            inputs = send_inputs_to_device(inputs, torch_device)
            loss, scores = bert_model(**inputs)[:2]
            validation_losses.append(loss.cpu().item())

            _, classifications = torch.max(scores, 1)
            preds.append(classifications.cpu().numpy())
        bert_model.train()
    return np.concatenate(preds), np.concatenate(labels)

epoch_bar = tqdm(range(epochs_iter))
loss_acc = 0
alpha = 0.95
for epoch in epoch_bar:
    batch_bar = tqdm(enumerate(train_data_loader), desc=f'Epoch {epoch}', total=len(train_data_loader))
    for idx, inputs in batch_bar:
        inputs = send_inputs_to_device(inputs, torch_device)
        optimizer.zero_grad()
        loss, logits = bert_model(**inputs)[:2]
        
        loss.backward()
        optimizer.step()
        
        # calculate a simplified ewma to the loss
        if epoch == 0 and idx == 0:
            loss_acc = loss.cpu().item()
        else:
            loss_acc = loss_acc * alpha + (1-alpha) * loss.cpu().item()
        
        batch_bar.set_postfix(loss=loss_acc)
        
        if idx%5000 == 0:
            preds, labels = predict(bert_model, val_data_loader, torch_device)
            metrics = compute_metrics(preds, labels)
            print(metrics)
            

        scheduler.step()
    os.makedirs('/kaggle/working/checkpoints/epoch'+str(epoch))
    bert_model.save_pretrained('/kaggle/working/checkpoints/epoch'+str(epoch))  