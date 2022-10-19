import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def tokenize_categories(postags, max_len):
    
    # Create tokenizer and fit
    tag_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="-UNK-")
    tag_tokenizer.fit_on_texts(postags)

    # Encode and pad
    y_encoded = tag_tokenizer.texts_to_sequences(postags)
    y = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=y_encoded, padding='post', truncating='post')

    # Turn to one-hot
    y_cat = to_categorical(y)
    n_classes = y_cat.shape[2]

    return y_cat, tag_tokenizer