import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def tokenize_categories(postags, max_len):
    '''
        inputs:
            postags: list of lists of postags
            max_len: padding value
        
        outputs:
            Y: sentences tokenize-encoded list of lists of postags
            tag_tokenizer: tokenizer employed
    '''

    # Create tokenizer and fit
    tag_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="-UNK-")
    tag_tokenizer.fit_on_texts(postags)
    y_encoded = tag_tokenizer.texts_to_sequences(postags)
    Y = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=y_encoded, padding='post', truncating='post')
    Y = to_categorical(Y)

    return Y, tag_tokenizer