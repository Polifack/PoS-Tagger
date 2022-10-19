def tokenize_categories(postags, max_len):
    
    # Create tokenizer and fit
    tag_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="-UNK-")
    tag_tokenizer.fit_on_texts(tags)

    # Encode and pad
    y_encoded = tag_tokenizer.texts_to_sequences(y)
    y = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=y_encoded, padding='post', truncating='post')

    # Turn to one-hot
    y_cat = to_categorical(y)
    n_classes = y_cat.shape[2]

    return y_cat, tag_tokenizer