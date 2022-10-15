from encs.dependency import encode_dependencies, decode_dependencies
from encs.constituent import encode_constituent, decode_constituent
from utils.constants import *

import argparse
import time

if __name__=="__main__":

    encodings = [C_ABSOLUTE_ENCODING, C_RELATIVE_ENCODING, C_DYNAMIC_ENCODING, 
                D_ABSOLUTE_ENCODING, D_RELATIVE_ENCODING, D_POS_ENCODING, D_BRACKET_ENCODING, D_BRACKET_ENCODING_2P]

    parser = argparse.ArgumentParser(description='Constituent and Dependencies Linearization System')
    parser.add_argument('operation', metavar='op',type=str, choices=[OP_TRAIN, OP_PRED],
                        help='Operation mode of the system.')

    parser.add_argument('input', metavar='in file', type=str,
                        help='Path of the .conllu file to train the model or text file containing sentences to predict')

    parser.add_argument('output', metavar='out file', type=str,
                        help='Path of the output folder where to store the model in training mode or path to store the predicted text file')

    parser.add_argument('--config', metavar="config", required=False, 
                        help='Path of the config file specifying the architecture of the sequence labeller')
            
    parser.add_argument("--time", action='store_true', required=False, 
                        help='Flag to measure decoding time.')

    parser.add_argument('--feats', metavar="features", nargs='+', default=None,
                        help='Train ONLY: Set of additional features to add to the training', required=False)

    args = parser.parse_args()

    if args.time:
        start_time=time.time()

    ##
    ## main operation
    if args.operation == OP_TRAIN:
        n_tokens, n_sentences = train
    
    elif args.operation == OP_PRED:
        n_tokens, n_sentences = predict(args.input, args.output)
    ##

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
