import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def tokenize_sentences(sentences, max_len):

    # Turn sentences into arrays of words
    x_words = [[w for w in s] for s in sentences]
    
    # Create tokenizer
    text_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<oov>")
    text_tokenizer.fit_on_texts(x_words)

    # Encode and pad
    x_words = text_tokenizer.texts_to_sequences(x_words)
    x_words = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=x_words, padding='post', truncating='post')

    return x_words, text_tokenizer