# -*- coding: utf-8 -*-
"""
@author: gkredila
"""

"""
-------------------------------------------------------------------------------
-- Import Packages --
-------------------------------------------------------------------------------
"""
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)

import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

def viz_wordcloud(df, col):
    port_stopwords = nltk.corpus.stopwords.words('portuguese')
    words = ' '.join([word for word in df[col]])
    wcloud = WordCloud(width=800, height=500, max_font_size=110,
                       collocations=False, stopwords = port_stopwords, 
                       background_color = 'white').generate(words)
    plt.figure(figsize=(10,7))
    plt.imshow(wcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wcloud.process_text(words), wcloud

def zip_wordcounts_relative_freq(keys_list, word_freq_list, rel_freq_list):
    zipped_items = list(zip(word_freq_list, rel_freq_list))
    freq_dict = {k:v for k,v in zip(keys_list, zipped_items)}
    
    return freq_dict

def print_word_rel_freq(dct, sentiment):
    print(f"Top 20 {sentiment} Words:")
    print("* Each word is followed by its word and relative frequency *\n")
    for word, freq in dct.items():
        print("{}:\t{}".format(word, freq))

# convert review scores to sentiment
def conv_to_sentiment(review_score):
    review_score = int(review_score)
    
    if review_score <= 2:
        return 'Negative'
        #return 0
    #elif review_score == 3:
     #   return 'Neutral'
        #return 2
    else:
        return 'Positive'
    
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
    train_dataset, test = train_test_split(data, test_size=test_dataset_size,
                                           random_state=808, 
                                           stratify=data['review_sentiment'])
    train, validation = train_test_split(train_dataset, 
                                         test_size=test_dataset_size, 
                                         random_state=808, 
                                         stratify=train_dataset['review_sentiment'])
    
    return train, validation, test

def create_tokenized_data(tokenizer, split_data_list):
    train_df, validation_df, test_df = split_data_list
    train_df.to_csv('train_df_split.csv', encoding = 'utf_8_sig')
    validation_df.to_csv('validation_df_split.csv', encoding = 'utf_8_sig')
    test_df.to_csv('test_df_split.csv', encoding = 'utf_8_sig')
    
    tokenized_dataset = datasets.load_dataset(
        'csv', 
        data_files={'train': 'train_df_split.csv',
                    'validation':'validation_df_split.csv',
                    'test': 'test_df_split.csv'})
    tokenized_dataset = tokenized_dataset.map(
        lambda example: {'pretok_text': re.sub(r':[\)\(]+',
                                               '',
                                               str(example['review_message']))},
        batched=False)
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: tokenizer(examples['pretok_text']), 
        batched=True)
    tokenized_dataset = tokenized_dataset.map(
        lambda example: {'labels': 1 if example['review_sentiment'] == 'Positive' else 0}, 
        batched=False)
    tokenized_dataset.set_format(type='torch', 
                                 columns=['input_ids', 
                                          'token_type_ids', 
                                          'attention_mask', 
                                          'labels'])
    
    return tokenized_dataset

"""
--------------------------------------------
-- Helpers
--------------------------------------------
"""
def evaluate_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                               predictions, 
                                                               average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def input_to_torch_device(inputs, torch_device):
    return {key:tensor.to(torch_device) for key, tensor in inputs.items()}


"""
--------------------------------------------
-- Eval Function
--------------------------------------------
"""
def eval_preds(bert_model, validation_data_loader, torch_device):
    with torch.no_grad():
        bert_model.eval()
        predictions = []
        labels = []
        validation_losses = []
        for inputs in validation_data_loader:
            labels.append(inputs['labels'].numpy())
            
            inputs = input_to_torch_device(inputs, torch_device)
            loss, scores = bert_model(**inputs)[:2]
            validation_losses.append(loss.cpu().item())

            _, classifications = torch.max(scores, 1)
            predictions.append(classifications.cpu().numpy())
        bert_model.train()
    return np.concatenate(predictions), np.concatenate(labels)

def viz_conf_matrix(confusion_matrix_dataframe):
    heat_map = sns.heatmap(confusion_matrix_dataframe, annot=True, fmt="d", 
                           cmap="crest", linewidth = 0.5)
    heat_map.yaxis.set_ticklabels(heat_map.yaxis.get_ticklabels(), 
                                  rotation=0, ha='right')
    heat_map.xaxis.set_ticklabels(heat_map.xaxis.get_ticklabels(), 
                                  rotation=0, ha='right')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

