# PoS-Tagger

Part of speech tagging system built in Keras and Python. The architecture was inspired by [Part-of-Speech Tagging with Bidirectional Long Short-Term Memory
Recurrent Neural Network](https://arxiv.org/pdf/1510.06168.pdf). This tagger is based on a sequence labeller model consisting on 3 layers:

1) Input layer: Word tokenization and embedding + Character tokenization and embedding.
2) Hidden layer: Bi-LSTM layer.
3) Inference layer: Dense with softmax (default) activation function.

## Usage:

In order to test the tagger we provide a ./install.sh file that will create the conda virtual enviroment and install all required python modules and a ./demo.sh file that will train a model using the UD_EWT_English treebank and decode a sample file. In this repo we provide a pre-trained model in the ./output folder and a sample file to decode in sample.txt. We already provide the predict results in predict.txt. 

### Train

The training of the tagger will require a folder containing the treebanks in conll-U format divided in train-validation-test splits named following the universal dependencies convention: train.conllu, test.conllu, dev.conllu. During the training process we can pass as arguments the following parameters (with the ones marked with * being mandatory):

- Input Directory*: Folder storing the dataset that will be employed for the test/validation/training processs.
- Output Directory*: Folder where the model will be saved.
- Sentence Length: Maximum sentence length that will be considered during training. Sentences excceeding this length will be padded out to this value. Default value is 128.
- Word Length: Maximum word length that will be considered during training. Words exceeding this length will be padded out to this value. Default value is 16.
- Hidden Dimension: Number of Bi-LSTM units in the hidden dimension of the model. Default value is 32.
- Character Hidden Dimension: Number of Bi-LSTM units in the word embedding module of the model. Default value is 16.
- Activation function: Activation function that will be used in the inference layer of the model. Currently, only softmax and reul are supported. Default is softmax.
- Loss function: Loss function employed during the train process. Currently only categorical crossentropy is supported. 
- Optimizer: Optimizer employed during the training. Default is adam.
- Learning rate: Learning rate employed by the optimizer. Default is 0.001.
- Batch Size: Number of elements in each training epochs. Default is 32.
- Word-embeddings only: Do not use character embedding layer. Default is False.
- Training epochs: Epochs employed during training. Default is 10.

The training process can be executed with:

````
python main.py 
                  train 
                  --input <dataset_directory> 
                  --output <save_model_dir> 
                  --sentl <sentence_length> 
                  --wordl <word_length> 
                  --hdin <hidden_dimension> 
                  --chdim <character_hidden_dimension> 
                  --activation [relu, softmax]
                  --loss [categorical_crossentropy]
                  --lr <learning_rate>
                  --optimizer [adam, sdg]
                  --bsize <training_batch_size>
                  --wonly <skip_char_emb_layer>
                  --epochs <epochs_training>
````

### Decode

In order to part of speech tags we will need (i) a folder containing the pre-trained models and the vocabulary-filled tokenizers (the program will produce those during the training step) and (ii) a file with a sentence per line. The decoding process will be executed with

````
python main.py
                  decode
                  --input <file_to_decode_path>
                  --model_dir <model_directory>
                  --output <tagged_file_path>
                  
````
## Results

Results of the training for the Universal Dependencies English Web Treebank

## Google colab demo

Google colab project explaining the system:

## Future work

- Implement vectorization as input layer.
- Add pre-trained word embeddings.
- Implement sparse_categorical_crossentropy as loss function and as option when removing the one-hot-encoding of the tags.
- Add more metrics than accuracy.
- Test how the embedding of more features work (i.e. position in sentence, word depth in constituent_tree...)
