#import unicodedata
import re
import random
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import string

max_seq_length = 400
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
scale_map = {1:1, 3:4, 4:4, 5:3, 6:3, 7:1/2.5, 8: 1/5}

def scale_score(row):
    return row['domain1_score']*scale_map[row['essay_set']]

def load_data(path):
    data = []
    data_df = pd.read_csv(path, sep='\t', encoding='latin1')
    data_df = data_df[data_df['essay_set']!=2]
    contents = data_df['essay']
    #data_df['score'] = data_df.apply(lambda x: scale_score(x), axis=1)
    data_df['score'] = data_df['domain1_score']
    labels = data_df['score'] 
    ids = data_df['essay_id']
    essay_set_ids = data_df['essay_set']
    for content, label, id, essay_set_id in zip(contents, labels, ids, essay_set_ids):
        example = {}
        example["text"] = content
        example['score'] = float(label)
        example['id'] = int(id)
        example['essay_set'] = int(essay_set_id)
        if example["score"] is None:
            continue        
        data.append(example)
    random.seed(1)
    random.shuffle(data)
    return data

def tokenize(text):
    text = text.encode('utf-8', 'ingore').decode('ascii', "ignore")
    #remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
    vocabulary = set([word for word in word_counter if word_counter[word] > 2])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary 
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    index_to_word = dict(zip(range(len(vocabulary)), vocabulary))

    return word_indices, index_to_word, len(vocabulary)

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """

    max_seq_length = 0
    for i, dataset in enumerate(datasets):
        for example in dataset:
            max_seq_length = max(max_seq_length, len(tokenize(example['text'])))

    #print("max_seq_length", max_seq_length)
    
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
            example['score'] = torch.FloatTensor([example['score']])

            