import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def tokenize_sentences(sentences, words, max_len):
    '''
        inputs:
            sentences: list of strings
            words: list of list of words corresponding to the strings of sentences
            max_len: padding value
        
        outputs:
            X: tokenize-encoded and padded words
            text_tokenizer: tokenizer employed
    '''

    text_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<oov>")
    text_tokenizer.fit_on_texts(sentences)
    x_encoded = text_tokenizer.texts_to_sequences(words)

    # Apply padding
    X = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=x_encoded, padding='post', truncating='post')

    return X, text_tokenizer