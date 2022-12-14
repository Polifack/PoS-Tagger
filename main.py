import os
# disable tensorflow cpu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from models.tagger import SeqTagger
from models.conll_tree import ConllTree

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import pickle

import argparse

# set padding value constant
PAD_VAL = 0

def save_tokenizer(path, tokenizer):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def tokenize_and_pad(tokenizer, content, padding):
    X = tokenizer.texts_to_sequences(content)
    X = keras.utils.pad_sequences(maxlen=padding, sequences=X, padding='post', truncating='post', value = PAD_VAL)
    return X
def tokenize_and_pad_chars(char_tokenizer, content, padding, padding_char):
    padding_vector = np.asarray(np.full(padding_char, PAD_VAL))
    X = []

    for sequence in content:
        x = []
        char_padded_sequence = tokenize_and_pad(char_tokenizer, sequence.split(" "), padding_char)
        
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
    
    parser.add_argument('operation', metavar='op',type=str, choices=["train", "decode", 'evaluate'],
                        help='Operation mode of the system.')

    parser.add_argument('--input', metavar='in file', type=str, required=True,
                        help='Path of the .conllu file to train the model or text file containing sentences to predict')

    parser.add_argument('--output', metavar='out file', type=str, required=True, default=None,
                        help='Path of the output folder where to store the model in training mode or path to store the predicted text file')

    parser.add_argument('--modeldir', metavar='sentence', type=str, required=False,
                        help='Saved model and dataset directory')

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
    
    parser.add_argument('--loss', metavar='loss', required=False, default='categorical_crossentropy', choices=['categorical_crossentropy'],
                        help='keras loss function')

    parser.add_argument('--lr', metavar='learning_rate', required=False, default=0.001, 
                        help='Learning rate for the model')
    
    parser.add_argument('--optimizer', metavar='optim', required=False, choices=['adam','sgd'], default='adam',
                        help='LR optimizer for the model')

    parser.add_argument('--bsize', metavar="batch_size", required=False, default=32,
                        help='Batch training size')

    parser.add_argument('--epochs', required=False, default=10, 
                        help='Number epochs training.')
    
    parser.add_argument('--dropout', required=False, default=0.5,
                        help='Drop-out layer size of the model.')

    args = parser.parse_args()

    return args

def train(args):
    print("--> Training mode")

    sentl = int(args.sentl)
    wordl = int(args.wordl)
    hdim  = int(args.hdim)
    chdim = int(args.chdim)

    # Read treebank and extract relevant information
    test  = ConllTree.read_conllu_file(args.input+"/test.conllu")
    dev   = ConllTree.read_conllu_file(args.input+"/dev.conllu")
    train = ConllTree.read_conllu_file(args.input+"/train.conllu")

    tr_sents   = [t.get_sentence() for t in train]
    tr_postags = [t.get_postags() for t in train]

    te_sents = [t.get_sentence() for t in test]
    te_postags = [t.get_postags() for t in test]

    dv_sents = [t.get_sentence() for t in dev]
    dv_postags = [t.get_postags() for t in dev]

    print("Total number of TRAIN tagged sentences: {}".format(len(tr_sents)))
    print("Total number of DEV tagged sentences: {}".format(len(dv_sents)))
    print("Total number of TEST tagged sentences: {}".format(len(te_sents)))

    # Categories tokenization and one-hot
    tag_tokenizer = keras.preprocessing.text.Tokenizer(filters="")
    tag_tokenizer.fit_on_texts(tr_postags)
    num_cats = (len(tag_tokenizer.word_index) + 1)
    print("[*] TAG vocabulary size =",num_cats)

    y_tr_a = tokenize_and_pad(tag_tokenizer, tr_postags, sentl)
    y_val_a = tokenize_and_pad(tag_tokenizer, dv_postags, sentl)
    y_te_a = tokenize_and_pad(tag_tokenizer, te_postags, sentl)

    y_tr = to_categorical(y_tr_a)
    y_val = to_categorical(y_val_a)
    y_te = to_categorical(y_te_a)

    # Text tokenization
    text_tokenizer = keras.preprocessing.text.Tokenizer(filters="", oov_token="<oov>")
    text_tokenizer.fit_on_texts(tr_sents+dv_sents)                   
    num_words = (len(text_tokenizer.word_index) + 1)
    print("[*] WORD vocabulary size =",num_words)

    x_words_tr  = tokenize_and_pad(text_tokenizer, tr_sents, sentl)
    x_words_val = tokenize_and_pad(text_tokenizer, dv_sents, sentl)
    x_words_te  = tokenize_and_pad(text_tokenizer, te_sents, sentl)

    # Character tokenization
    char_tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, filters="")
    char_tokenizer.fit_on_texts(tr_sents+dv_sents)
    num_chars = (len(char_tokenizer.word_index) + 1)
    print("[*] CHAR vocabulary size =",num_chars)

    x_chars_tr = tokenize_and_pad_chars(char_tokenizer, tr_sents, sentl, wordl)
    x_chars_val = tokenize_and_pad_chars(char_tokenizer, dv_sents, sentl, wordl)
    x_chars_te = tokenize_and_pad_chars(char_tokenizer, te_sents, sentl, wordl)
    
    # Create model
    print("[*] Creating seq2seq model...")
    tagger = SeqTagger(num_cats, num_words, sentl, wordl, hdim, args.activation, args.dropout, num_chars, chdim)

    x_tr = [x_words_tr, x_chars_tr]
    x_val = [x_words_val, x_chars_val]

    tagger.build_model()
    
    learning_rate = float(args.lr)
    tagger.compile_model(args.loss, args.optimizer, learning_rate)
    tagger.show_model()

    bsize = int(args.bsize)
    epochs = int(args.epochs)
    tagger.train_model(x_tr, y_tr, bsize, epochs, x_val, y_val)

    # Save the model and the tokenizer
    tagger.save_model(args.output)
    save_tokenizer(args.output+"/words_tok.tokenizer", text_tokenizer)
    save_tokenizer(args.output+"/tags_tok.tokenizer", tag_tokenizer)
    save_tokenizer(args.output+"/chars_tok.tokenizer", char_tokenizer)

def evaluate(args):
    # read model from file
    print("--> Evaluation mode")
    tagger = SeqTagger()
    tagger.load_model(args.input)

    # plot model and save it
    tagger.plot_history(True, args.output)

def decode(args):
    print("--> Decoding mode")

    sentence_file = open(args.input)
    sentences = sentence_file.readlines()

    # load models and tokenizers
    model = tf.keras.models.load_model(args.modeldir+'/model.h5')    
    model.summary()

    text_tokenizer = load_tokenizer(args.modeldir+'/words_tok.tokenizer')     
    tag_tokenizer = load_tokenizer(args.modeldir+'/tags_tok.tokenizer')

    char_tokenizer = load_tokenizer(args.modeldir+'/chars_tok.tokenizer')
    x_words = tokenize_and_pad(text_tokenizer, sentences, args.sentl)
    x_chars = tokenize_and_pad_chars(char_tokenizer, sentences, args.sentl, args.wordl)
        
    x=[x_words, x_chars]

    y = model.predict(x)

    with open(args.output, "w") as out_file:
        for sentence, pred_tags in zip(sentences,y) :
            n_words = len(sentence.split(" "))
            unpadded_tags = pred_tags[0:n_words, :]
            tokenized_tags = np.argmax(unpadded_tags, axis=1)
            tags = tag_tokenizer.sequences_to_texts([tokenized_tags])[0]

            write_line = sentence+'\n'+tags+"\n"
            out_file.write(write_line)


if __name__=="__main__":
    args=parse_args()

    if args.operation == 'train':
        train(args)
    elif args.operation == 'decode':
        decode(args)
    elif args.operation == 'evaluate':
        evaluate(args)
