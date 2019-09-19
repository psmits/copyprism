# from google.colab import drive
# drive.mount('/content/drive')
import nltk
# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
# !cp drive/My\ Drive/ikea.csv .
# import numpy as np
# from numpy import array
# from pickle import dump
import pandas as pd
from sklearn.model_selection import train_test_split
# import sys
import string
from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Embedding
# from keras.utils import np_utils, to_categorical
# from keras.callbacks import ModelCheckpoint
# from keras.preprocessing.text import Tokenizer

nltk.download('stopwords')


def clean_text(input):
    # lowercase everything to standardize it
    input = input.lower()

    # tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove non alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


# save tokens to file, one sequence per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


ikea_items = pd.read_csv('ikea.csv')

# some items do not have descriptions from the specific box
ikea_items = ikea_items.dropna()

# some descriptions are identical
desc_uni = ikea_items.drop_duplicates(subset='description')

# average description length for future generation
# desc_avg = round(sum( map(len, desc_uni) ) / len(desc_uni))
# desc_std = map(len, desc_uni).std()

# split train and test
desc_train, desc_test = train_test_split(desc_uni, test_size=0.2)
pd.DataFrame(desc_train).to_csv('ikea_train.csv')
pd.DataFrame(desc_test).to_csv('ikea_test.csv')

# make one corpus
desc_single = ' '.join(desc_train.description)

# !cp ikea_test.csv drive/My\ Drive/.
# !cp ikea_train.csv drive/My\ Drive/.

tokens = clean_text(desc_single)

# print('Total Tokens: %d' % len(tokens))
# print('Unique Tokens: %d' % len(set(tokens)))

# make sequences of words from the full corpus
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)


# print('Total Sequences: %d' % len(sequences))
out_filename = 'ikea_train_sequences.txt'
save_doc(sequences, out_filename)
