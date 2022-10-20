# PoS-Tagger

Part of speech tagging system built in Keras and Python. This tagger is based on a sequence labeller model consisting on 3 layers:

1) Input layer: Word tokenization and embedding + Character tokenization and embedding
2) Hidden layer: Bi-LSTM 
3) Inference layer: Dense with softmax (default) activation function

Tested with the UD_EWT_English treebank but can read any CONLL-u file as input.

## Usage:

### Train

````
python main.py 
                  train 
                  <in_dir> 
                  <out_dir> 
                  --sentl <sentence_length> 
                  --wordl <word_length> 
                  --hdin <hidden_dimension> 
                  --chdim <character_hidden_dimension> 
                  --activation <activation_function>
                  --bsize <training_batch_size>
                  --wonly <skip_char_emb_layer>
                  --epochs <epochs_training>
````

### Decode

Coming soon

### To Do

- Implement vectorization as input layer
- Add pre-trained word embeddings
- 
