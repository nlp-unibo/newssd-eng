import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


def prepare_data(X, tokenizer, maxSentenceLen, y=[]):
    pad = tf.keras.preprocessing.sequence.pad_sequences  # (seq, padding = 'post', maxlen = maxlen)
    tokenizer = tokenizer
    dataFields = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "Label": []
    }
    lbls = {
        'SUBJ': 1.0,
        'OBJ': 0.0
    }
    for i in range(len(X)):
        data = tokenizer(X[i])
        padded = pad([data['input_ids'], data['attention_mask'], data['token_type_ids']], padding='post',
                     maxlen=maxSentenceLen)
        dataFields['input_ids'].append(padded[0])
        dataFields['attention_mask'].append(padded[1])
        dataFields['token_type_ids'].append(padded[-1])
    if len(y):
        dataFields['Label'] = list(map(lambda e: lbls[e], y))

    for key in dataFields:
        dataFields[key] = np.array(dataFields[key])

    return [dataFields["input_ids"], dataFields["token_type_ids"], dataFields["attention_mask"]], dataFields["Label"]


def toLabels(data, subT=0.5):
    ypred = []
    for pred in data:
        if pred >= subT:
            ypred.append('SUBJ')
        else:
            ypred.append('OBJ')
    return ypred


def read_data(path, split):
    df = pd.read_csv(os.path.join(path, f"{split}.csv"))
    return df["Sentence"].values, df["Label"].values


def get_shuffled(x, y):
    data = list(zip(x, y))
    random.shuffle(data)
    return zip(*data)


def pick_configuration(train_conf="en", test_conf="en"):
    if train_conf == "en":
        x_train, y_train = read_data("data/english", "train")
    elif train_conf == "it":
        x_train, y_train = read_data("data/italian", "train")
    elif train_conf == "en+it":
        x_train_en, y_train_en = read_data("data/english", "train")
        x_train_it, y_train_it = read_data("data/italian", "train")
        x_train, y_train = get_shuffled(x=list(x_train_en) + list(x_train_it), y=list(y_train_en) + list(y_train_it))
    else:
        raise Exception("Wrong train conf requested")
    if test_conf == "en":
        x_test, y_test = read_data("data/english", "test")
        x_val, y_val = read_data("data/english", "val")
    elif test_conf == "it":
        x_test, y_test = read_data("data/italian", "test")
        x_val, y_val = read_data("data/italian", "val")
    else:
        raise Exception("Wrong test conf requested")
    return x_train, y_train, x_test, y_test, x_val, y_val
