def create_text_vector_layer(sentences, max_len):

    # create vectorizer
    text_vectorizer = keras.layers.TextVectorization(output_mode="int", output_sequence_length=max_len)

    # adapt
    text_vectorizer.adapt(sentences)