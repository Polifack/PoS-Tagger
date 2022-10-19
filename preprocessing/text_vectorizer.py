import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def create_text_vector_layer(sentences, max_len):

    # create vectorizer
    text_vectorizer = keras.layers.TextVectorization(output_mode="int", output_sequence_length=max_len)

    # adapt
    text_vectorizer.adapt(sentences)