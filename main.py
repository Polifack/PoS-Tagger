import os
# disable tensorflow cpu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from models.tagger import SeqTagger

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils.np_utils import to_categorical
from conllu import parse_tree_incr
import matplotlib.pyplot as plt
import pickle

import argparse
import time


def parse_conllu(in_path):
    sentences = []
    postags = [] 
    words = []

    in_file = open(in_path)

    for i, token_tree in enumerate(parse_tree_incr(in_file)):
        print("[*] Reading "+in_path+":"+str(i), end="\r")

        current_tags = []
        current_words = []
        
        data = token_tree.serialize().split('\n')
        dependency_start_idx = 0
        # skip info lines
        for line in data:
            if line[0]!="#":
                break
            if "# text" in line:
                sentence = line.split("# text = ")[1]
                sentences.append(sentence)
            dependency_start_idx+=1
        
        data = data[dependency_start_idx:]

        # data lines
        for line in data:
            # check if not valid line
            if (len(line)<=1) or len(line.split('\t'))<10 or line[0] == "#":
                continue
            wid,form,lemma,upos,xpos,feats,head,deprel,deps,misc = line.split('\t')
            current_tags.append(upos)
            current_words.append(form)

        postags.append(current_tags)
        words.append(current_words)    
    
    print("[*] Reading "+in_path+": Done")
    return sentences, postags, words
def train_split_reader(in_path):
    fpath_train = in_path+"/train.conllu"
    fpath_test = in_path+"/test.conllu"
    fpath_dev = in_path+"/dev.conllu"

    return (parse_conllu(fpath_train), parse_conllu(fpath_test), parse_conllu(fpath_dev))

def save_tokenizer(path, tokenizer):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
def tokenize_and_pad(tokenizer, content, padding):
    X = tokenizer.texts_to_sequences(content)
    X = keras.preprocessing.sequence.pad_sequences(maxlen=padding, sequences=X, padding='post', truncating='post')
    return X
def tokenize_and_pad_chars(char_tokenizer, content, padding, padding_char):
    padding_vector = np.asarray(np.full(padding_char, 0))
    X = []

    for sequence in content:
        x = []
        char_padded_sequence = tokenize_and_pad(char_tokenizer, sequence, padding_char)
        
        if len(char_padded_sequence)>=padding:
            # trim
            x = char_padded_sequence[:padding]
        else:
            # pad
            x = char_padded_sequence

            for i in range(len(char_padded_sequence), padding):
                # add columns below
                x=np.vstack((x, padding_vector))

        X.append(np.array(x))
        
    return np.asarray(X)
def parse_args():

    parser = argparse.ArgumentParser(description='PoS tagger')
    
    parser.add_argument('operation', metavar='op',type=str, choices=["train", "predict"],
                        help='Operation mode of the system.')

    parser.add_argument('input', metavar='in file', type=str,
                        help='Path of the .conllu file to train the model or text file containing sentences to predict')

    parser.add_argument('output', metavar='out file', type=str,
                        help='Path of the output folder where to store the model in training mode or path to store the predicted text file')

    parser.add_argument('--sentl', metavar="sent_length", required=False, default=128, 
                        help="Max length of sentence allowed")

    parser.add_argument('--wordl', metavar="word_length", required=False, default=16,
                        help="Max length of words allowed")

    parser.add_argument('--hdim', metavar='hidden_dimension',required=False,default=32,
                        help='Hidden dimension of the sequence labeling system.')

    parser.add_argument('--chdim', metavar='hidden_dimension_char',required=False,default=16,
                        help='Hidden dimension of the character embedding.')
    
    parser.add_argument('--activation', metavar='activation', required=False, choices=['relu', 'softmax'], default='softmax',
                        help='Activation function for the inference layer of the sequence labeling system')
    
    parser.add_argument('--bsize', metavar="batch_size", required=False, default=32,
                        help='Batch training size')

    parser.add_argument('--charemb', required=False, default=True, action='store_true', 
                        help='Use character embeddings layer.')

    parser.add_argument('--epochs', required=False, default=10, 
                        help='Number epochs training.')

    args = parser.parse_args()
    return args



if __name__=="__main__":
    args=parse_args()

    # Read treebank and extract relevant information
    (train, test, dev) = train_split_reader(args.input)
    tr_sents, tr_postags, tr_words = train
    te_sents, te_postags, te_words = test
    dv_sents, dv_postags, dv_words = dev
    print("Total number of tagged sentences: {}".format(len(tr_sents)))

    # Categories tokenization and one-hot
    tag_tokenizer = keras.preprocessing.text.Tokenizer()
    tag_tokenizer.fit_on_texts(tr_postags)
    num_cats = (len(tag_tokenizer.word_index) + 1)
    print("[*] TAG vocabulary size =",num_cats)
    
    y_tr_a = tokenize_and_pad(tag_tokenizer, tr_postags, args.sentl)
    y_val_a = tokenize_and_pad(tag_tokenizer, dv_postags, args.sentl)
    y_te_a = tokenize_and_pad(tag_tokenizer, te_postags, args.sentl)
    
    y_tr = to_categorical(y_tr_a)
    y_val = to_categorical(y_val_a)
    y_te = to_categorical(y_te_a)

    # Text tokenization
    text_tokenizer = keras.preprocessing.text.Tokenizer()
    text_tokenizer.fit_on_texts(tr_words+te_words+dv_words)                   ## should the tokenizer be fitted with train+dev+test?
    num_words = (len(text_tokenizer.word_index) + 1)
    print("[*] WORD vocabulary size =",num_words)

    x_words_tr = tokenize_and_pad(text_tokenizer, tr_words, args.sentl)
    x_words_val = tokenize_and_pad(text_tokenizer, dv_words, args.sentl)
    x_words_te = tokenize_and_pad(text_tokenizer, te_words, args.sentl)

    # Character tokenization
    if args.charemb:
        char_tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        char_tokenizer.fit_on_texts(tr_words+te_words+dv_words)
        num_chars = (len(char_tokenizer.word_index) + 1)
        print("[*] CHAR vocabulary size =",num_chars)
        
        x_chars_tr = tokenize_and_pad_chars(char_tokenizer, tr_words, args.sentl, args.wordl)
        x_chars_val = tokenize_and_pad_chars(char_tokenizer, dv_sents, args.sentl, args.wordl)
        x_chars_te = tokenize_and_pad_chars(char_tokenizer, te_sents, args.sentl, args.wordl)

    # Create model
    print("[*] Creating seq2seq model...")
    tagger = SeqTagger(num_cats, num_words, num_chars, args.sentl, args.wordl, args.hdim, args.activation, 
                        char_embs=args.charemb, char_hidden_dim=args.chdim)
    tagger.build_model()
    tagger.compile_model()
    tagger.show_model()
    tagger.train_model([x_words_tr, x_chars_tr], y_tr, args.bsize, args.epochs, [x_words_val, x_chars_val], y_val)

    # Save the model and the tokenizer
    tagger.save_model(args.output)
    save_tokenizer(args.output+"/word_tok.tokenizer", text_tokenizer)
    save_tokenizer(args.output+"/tags_tok.tokenizer", tag_tokenizer)
    save_tokenizer(args.output+"/chars_tok.tokenizer", char_tokenizer)
