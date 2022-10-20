import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def tokenize_characters(sentences, max_len, max_len_char):
    '''
        inputs:
            sentences: list of strings
            max_len: padding value sentence level
            max_len: padding value word level
        
        outputs:
            X_char: sentences character-level tokenize-encoded and padded
            char_tokenizer: tokenizer employed
    '''

    char_tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    char_tokenizer.fit_on_texts(sentences)
    
    X_char = []
    for sentence in sentences:
        sent_seq = []
        
        # get sentence words
        words = sentence.split(" ")
        
        # tokenize the word and pad
        encoded_words = char_tokenizer.texts_to_sequences(words)
        padded_words = keras.preprocessing.sequence.pad_sequences(maxlen=max_len_char, sequences=encoded_words, padding='post', truncating='post', value = 0)
        
        # append character-level tokenized words to the sentence
        for word in padded_words:
            sent_seq.append(word)

        # apply hand-crafted padding to sequence
        for i in range(len(words), max_len):
            sent_seq.append(np.full(max_len_char, 0))

        X_char.append(np.array(sent_seq))
    X_char=np.asarray(X_char)
    
    return X_char, char_tokenizer