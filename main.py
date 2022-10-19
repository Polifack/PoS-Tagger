from utilities.reader import * 
from preprocessing.categorize_tokenizer import tokenize_categories
from preprocessing.text_tokenizer import tokenize_sentences
from preprocessing.text_vectorizer import create_text_vector_layer
from preprocessing.char_indexer import index_characters
from models.tagger import SeqTagger

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

    parser.add_argument('--wordl', metavar="word_length", required=False, default=20,
                        help="Max length of words allowed")

    parser.add_argument('--wmode', metavar="word_mode", required=False, choices=['vec','tok'], default='tok',
                        help='Pre-processing mode of the input text. Could be vectorizer (vec) or tokenizer (tok).')

    parser.add_argument('--charemb', required=False, default=True, action='store_true', 
                        help='Use character embeddings layer')
            
    parser.add_argument("--time", action='store_true', required=False, 
                        help='Flag to measure decoding time.')

    args = parser.parse_args()
    print(args)

    if args.time:
        start_time=time.time()
    
    print("+-------------------------+")
    # print data about system


    # Read treebank and extract relevant information
    treebank = parse_conllu(args.input)

    print("[*] Extracting sentences... ",end="\r")
    sentences = [dt.get_sentence() for dt in treebank]
    print("[*] Extracting sentences: Done")


    print("[*] Extracting upos... ",end="\r")
    postags = [dt.get_upos() for dt in treebank]
    print("[*] Extracting upos: Done")
   
    words = set([word for sentence in sentences for word in sentence.split(" ")])
    n_words = len(words)
    print("[*] Number of words:",n_words)

    tags = set([word for sentence in postags for word in sentence])
    n_tags = len(tags)
    print("[*] Number of tags:",n_tags)

    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)
    print("[*] Number of chars:",n_chars)

    # Pre-process data
    print("[*] Converting upos to categories... ",end="\r")
    y, cat_tokenizer = tokenize_categories(postags, args.sentl)
    print("[*] Converting upos to categories: Done")


    if args.wmode == 'tok':
        print("[*] Tokenizing input sentences... ",end="\r")
        x_words, words_tokenizer = tokenize_sentences(sentences, args.sentl)
        print("[*] Tokenizing input sentences: Done")

    elif args.wmode =='vec':
        print("[*] Creating vector layer... ",end="\r")
        word_vectorizer = create_text_vector_layer(sentences, args.sentl)
        print("[*] Creating vector layer: Done")
    
    if args.charemb:
        print("[*] Setting characters as indexes... ",end="\r")
        x_chars, char2idx = index_characters(sentences, chars, args.sentl, args.wordl)
        print("[*] Setting characters as indexes: Done")


    # Create model
    print("[*] Creating seq2seq model...")


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
