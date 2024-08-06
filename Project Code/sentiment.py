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
sns.set_theme(style='whitegrid', font_scale=1.2, font='sans-serif')
sns.set_palette(sns.color_palette("Set2"))
import matplotlib.pyplot as plt
import re
import torch
import nltk

# if the console cannot find this file, please execute the file once
from postgresql_connection import db_connect
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix)
from sklearn.naive_bayes import MultinomialNB
from sentiment_data_processing import (conv_to_sentiment, regex_linebreaks, 
                                       regex_hyperlinks, regex_dates, 
                                       regex_money, regex_numbers, 
                                       regex_special_chars, regex_whitespace, 
                                       split_data, create_tokenized_data, 
                                       evaluate_metrics, input_to_torch_device, 
                                       eval_preds, viz_conf_matrix, 
                                       zip_wordcounts_relative_freq,
                                       print_word_rel_freq,
                                       viz_wordcloud)
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)


"""
-------------------------------------------------------------------------------
-- Load Data --
-------------------------------------------------------------------------------
"""

connection = db_connect()

#customer = pd.read_sql_table('customer', connection)
#geolocation = pd.read_sql_table('geolocation', connection)
#geolocation_state = pd.read_sql_table('geolocation_state', connection)
#order_item = pd.read_sql_table('order_item', connection)
#orders = pd.read_sql_table('orders', connection)
#product = pd.read_sql_table('product', connection)
#product_category = pd.read_sql_table('product_category', connection)
review = pd.read_sql_table('review', connection)
#seller = pd.read_sql_table('seller', connection)


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
# update data type of the review score field
review['review_score'] = review['review_score'].astype('int64').astype('category') 

# plot counts by score
fig_scores = sns.countplot(data = review, x="review_score", palette="Set2")
plt.xlabel("Review Score")
plt.title("Review Counts by Score")
plt.show(fig_scores)

# create new field sentiment for scores 3+ to positive sentiment, and <3 to negative sentiment
review['review_sentiment'] = review.review_score.apply(conv_to_sentiment)

# plot counts by sentiment
sentiment_classes = ['positive', 'negative']
fig_sentiment = sns.countplot(data = review, x="review_sentiment", palette="Set2").set_xticklabels(sentiment_classes)
plt.xlabel("Review Sentiment")
plt.title("Review Counts by Sentiment Type")
plt.show(fig_sentiment)

# remove reviews with Not Defined in the review message
review_new = review[review.review_message != 'Not Defined']

# visualize word cloud of positive and negative reviews
positive_wc_dict, positive_wc = viz_wordcloud(review_new.loc[review_new['review_sentiment'] == 'Positive'], 
              "review_message")
negative_wc_dict, negative_wc = viz_wordcloud(review_new.loc[review_new['review_sentiment'] == 'Negative'], 
              "review_message")
 
# visualize review score = 3 (neutral for interpreting results)
neutral_wc_dict, neutral_wc = viz_wordcloud(review_new.loc[review_new['review_score'] == 3], 
              "review_message")
# Sort the neutral dictionary
neutral_word_freq = {k: v for k, v in sorted(neutral_wc_dict.items(),
                                   reverse=True, 
                                   key=lambda item: item[1])}

# Print relative word frequencies
neutral_rel_freq = neutral_wc.words_
neutral_keys_top20 = list(neutral_word_freq.keys())[:20]
neutral_items_top20_word = list(neutral_word_freq.values())[:20]
neutral_items_top20_rel = list(neutral_rel_freq.values())[:20]
neutral_freq_dict = zip_wordcounts_relative_freq(neutral_keys_top20,
                                              neutral_items_top20_word,
                                              neutral_items_top20_rel)
print_word_rel_freq(neutral_freq_dict, "Neutral")
"""
Top 20 Neutral Words:
* Each word is followed by its word and relative frequency *

produto:	(1677, 1.0)
veio:       (540, 0.3220035778175313)
recebi:     (512, 0.3053070960047704)
entrega:    (478, 0.28503279666070364)
prazo:      (469, 0.2796660703637448)
bom:        (378, 0.22540250447227192)
chegou:     (349, 0.2081097197376267)
entregue:	(298, 0.17769827072152652)
comprei:	(287, 0.17113893858079904)
ainda:      (260, 0.15503875968992248)
porém:      (257, 0.15324985092426952)
bem:        (206, 0.12283840190816935)
correio:	(198, 0.11806797853309481)
dia:        (190, 0.11329755515802027)
compra:     (184, 0.10971973762671437)
loja:       (182, 0.10852713178294573)
qualidade:	(175, 0.10435301132975551)
nao:        (175, 0.10435301132975551)
gostei:     (169, 0.10077519379844961)
ante:       (152, 0.09063804412641623)
"""

"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Data Preprocessing --
-------------------------------------------------------------------------------
"""
# regex data preprocessing
review_new['review_message'] = regex_linebreaks(review_new['review_message'])
review_new['review_message'] = regex_hyperlinks(review_new['review_message'])
review_new['review_message'] = regex_dates(review_new['review_message'])
review_new['review_message'] = regex_money(review_new['review_message'])
review_new['review_message'] = regex_numbers(review_new['review_message'])
review_new['review_message'] = regex_special_chars(review_new['review_message'])
review_new['review_message'] = regex_whitespace(review_new['review_message'])

review_new.info()
"""
<class 'pandas.core.frame.DataFrame'>
Index: 39094 entries, 0 to 43062
Data columns (total 8 columns):
 #   Column               Non-Null Count  Dtype         
---  ------               --------------  -----         
 0   review_id            39094 non-null  object        
 1   order_id             39094 non-null  object        
 2   review_score         39094 non-null  category      
 3   review_title         39094 non-null  object        
 4   review_message       39094 non-null  object        
 5   review_dtm           39094 non-null  datetime64[ns]
 6   review_response_dtm  39094 non-null  datetime64[ns]
 7   review_sentiment     39094 non-null  object        
dtypes: category(1), datetime64[ns](2), object(5)
memory usage: 2.4+ MB
"""
sentiment_type = review_new.review_sentiment.value_counts()
sentiment_type
"""
review_sentiment
Positive    29811
Negative     9283
Name: count, dtype: int64
"""

# create tokenizer for preparing model inputs using the base architecture that 
# BERT has been trained on with Portuguese language support
bert_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')


"""
-------------------------------------------------------------------------------
-- Sentiment Analysis - Split Data and Prepare Torch Dataset --
-------------------------------------------------------------------------------
"""

# split into 80-10-10
train_data, validation_data, test_data = split_data(review_new)

print('Training Set:    ', train_data.shape[0])
print('Validation Set:  ', validation_data.shape[0])
print('Test Set:        ', test_data.shape[0])
"""
Training Set:     31276
Validation Set:   3909
Test Set:         3909
"""

tokenized_dict = create_tokenized_data(bert_tokenizer, (train_data, 
                                                        validation_data, 
                                                        test_data))


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
            predictions, labels = eval_preds(bert_model, 
                                             validation_data_loader, 
                                             torch_device)
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
test_predictions, test_labels = eval_preds(bert_model, test_data_loader, 
                                           torch_device)
test_metrics = evaluate_metrics(test_predictions, test_labels)
print(test_metrics)

transpose_sentiments = ['negative', 'positive']
print(classification_report(test_labels, test_predictions, 
                            target_names=transpose_sentiments))
"""
1 MIN 30 SEC for EPOCH 0
              precision    recall  f1-score   support

    negative       0.74      0.87      0.80       928
    positive       0.96      0.91      0.93      2981

    accuracy                           0.90      3909
   macro avg       0.85      0.89      0.87      3909
weighted avg       0.91      0.90      0.90      3909

VS 2 EPOCHS - Slight improvement (1 hr)

              precision    recall  f1-score   support

    negative       0.79      0.83      0.81       928
    positive       0.95      0.93      0.94      2981

    accuracy                           0.91      3909
   macro avg       0.87      0.88      0.87      3909
weighted avg       0.91      0.91      0.91      3909
"""

conf_mat = confusion_matrix(test_labels, test_predictions)
conf_mat_df = pd.DataFrame(conf_mat, index = transpose_sentiments, 
                           columns = transpose_sentiments)
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

log_reg = LogisticRegression(random_state=0, class_weight='balanced', 
                             max_iter=500, verbose=True)
log_reg.fit(xtrain_comp_data, ytrain_comp_data)
ypredictions_lr = log_reg.predict(xvalid_comp_data)

print(classification_report(yvalid_comp_data, ypredictions_lr, 
                            target_names=transpose_sentiments))
"""
              precision    recall  f1-score   support

    negative       0.69      0.92      0.79       928
    positive       0.97      0.87      0.92      2981

    accuracy                           0.88      3909
   macro avg       0.83      0.89      0.85      3909
weighted avg       0.90      0.88      0.89      3909
"""

conf_mat_lr = confusion_matrix(yvalid_comp_data, ypredictions_lr)
conf_mat_df_lr = pd.DataFrame(conf_mat_lr, index = transpose_sentiments, 
                              columns = transpose_sentiments)
viz_conf_matrix(conf_mat_df_lr)

# Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(xtrain_comp_data, ytrain_comp_data)
ypredictions_nb = naive_bayes.predict(xvalid_comp_data)
print(classification_report(yvalid_comp_data, ypredictions_nb, 
                            target_names=transpose_sentiments))
"""
              precision    recall  f1-score   support

    negative       0.74      0.76      0.75       928
    positive       0.92      0.92      0.92      2981

    accuracy                           0.88      3909
   macro avg       0.83      0.84      0.84      3909
weighted avg       0.88      0.88      0.88      3909
"""

conf_mat_nb = confusion_matrix(yvalid_comp_data, ypredictions_nb)
conf_mat_df_nb = pd.DataFrame(conf_mat_nb, index = transpose_sentiments, 
                              columns = transpose_sentiments)
viz_conf_matrix(conf_mat_df_nb)
