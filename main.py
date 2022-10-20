import os
# disable tensorflow cpu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from utilities.reader import * 
from preprocessing.categorize_tokenizer import tokenize_categories
from preprocessing.text_tokenizer import tokenize_sentences
from preprocessing.text_vectorizer import create_text_vector_layer
from preprocessing.char_tokenizer import tokenize_characters
from preprocessing.char_indexer import index_characters
from models.tagger import SeqTagger

import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import argparse
import time

if __name__=="__main__":

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

    parser.add_argument('--epochs', required=False, default=20, 
                        help='Number epochs training.')
            
    parser.add_argument("--time", action='store_true', required=False, 
                        help='Flag to measure decoding time.')

    args = parser.parse_args()
    print(args)

    if args.time:
        start_time=time.time()
    

    # Read treebank and extract relevant information
    # sentences = array of sentences
    # postags = array of array of tags
    # words = array of array of words
    sentences, postags, words = parse_conllu(args.input)
    print("Total number of tagged sentences: {}".format(len(sentences)))

    # Pre-process data
    y, cat_tokenizer = tokenize_categories(postags, args.sentl)
    num_cats = (len(cat_tokenizer.word_index) + 1)
    print("[*] TAG vocabulary size =",num_cats)

    x_words, words_tokenizer = tokenize_sentences(sentences, words, args.sentl)
    num_words = (len(words_tokenizer.word_index) + 1)
    print("[*] WORD vocabulary size =",num_words)

    if args.charemb:
        x_char, char_tokenizer = tokenize_characters(sentences, args.sentl, args.wordl)
        num_chars = (len(char_tokenizer.word_index) + 1)
        print("[*] CHAR vocabulary size =",num_chars)


    # get train/test split
    x_word_tr, x_word_te, y_tr, y_te = train_test_split(x_words, y, test_size=0.1, random_state=1)
    x_char_tr, x_char_te, _, _ = train_test_split(x_char, y, test_size=0.1, random_state=1)

    # Create model
    print("[*] Creating seq2seq model...")
    tagger = SeqTagger(num_cats, num_words, num_chars, args.sentl, args.wordl, args.hdim, args.activation, 
                        char_embs=args.charemb, char_hidden_dim=args.chdim)
    tagger.build_model()
    tagger.compile_model()
    tagger.show_model()
    tagger.train_model([x_word_tr, x_char_tr], y_tr, args.bsize, args.epochs, 0.1)

    if args.time:
        delta_time=time.time()-start_time
        fn_str=args.input.split("/")[-1]
        
        t_str="{:.2f}".format(delta_time)
        tt_str="{:2f}".format(delta_time/n_tokens)
        ts_str="{:2f}".format(delta_time/n_sentence)


        print("-----------------------------------------")
        print(fn_str+'@'+args.enc+':')
        print("-----------------------------------------")
        print('%10s' % ('total sentences'),n_trees)
        print('%10s' % ('total words'),n_labels)
        print('%10s' % ('total time'),t_str)
        print('%10s' % ('time per sentencel'),ts_str)
        print('%10s' % ('time per word'),tt_str)
        print("-----------------------------------------")
