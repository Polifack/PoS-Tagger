
def character_indexer(sentences, chars, max_len, max_len_char):
    # create character to index dictionary
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    # Create the array of sentences with indexes of char
    # As this needs two levels of padding (first to max_len_char and after to max_len)
    # this will be handcrafted

    x_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    next_char = char2idx.get(sentence[i][0][j])
                    word_seq.append(next_char)
                except:
                    next_char = char2idx.get("PAD")
                    word_seq.append(next_char)
            sent_seq.append(word_seq)
        x_char.append(np.array(sent_seq))

    return np.asarray(x), char2idx
    