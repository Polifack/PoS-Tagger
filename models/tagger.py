import tensorflow as tf
import keras

class SeqTagger:

    def __init__(self, n_cats, n_words, n_chars, max_sent_length, max_word_length, hidden_dim, activation, char_embs=False, char_hidden_dim=-1, seg_mode='tok', vec_layer=None):
        # Dataset settings
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.n_cats = n_cats
        self.n_words = n_words
        self.n_chars = n_chars
        
        # Obtain data from the tagger architecture
        self.hidden_dim = hidden_dim
        self.char_embs = char_embs
        self.char_hidden_dim = char_hidden_dim
        self.segmentation_mode = seg_mode
        self.activation_function = activation

        # Set segmentation words input mode
        self.seg_mode=seg_mode
        self.vec_layer=vec_layer
        

        self.model = None
        self.history = None

    def build_model(self):
        
        ''' Build the neural tagger according to the specified config in the initialization'''
        print("[*] Building model")

        # create input words layer
        if self.seg_mode == 'tok':
            print("[*] Creating word input layer; Input mode: Tokenization")
            word_in = layers.Input(shape=(self.max_sent_length,))

            emb_word = layers.Embedding(input_dim=self.n_words + 2, output_dim=self.hidden_dim,
                                input_length=self.max_sent_length)(word_in)
        elif self.seg_mode == 'vec':
            print("[*] Creating word input layer; Input mode: Vectorization")
            word_in = layers.Input(shape=(1,), dtype=tf.string)

            emb_word = self.vec_layer(word_in)
            emb_word = tf.keras.layers.Embedding(input_dim=self.n_words + 2, output_dim=self.hidden_dim,
                                input_length=self.max_sent_length)(emb_word)

        # Set input to the hidden layer (depends on character embeddings)
        if self.char_embs:
            print("[*] Creating character embedding layer")
            char_in = layers.Input(shape=(self.max_sent_length, self.max_word_length,))
            
            emb_char = layers.TimeDistributed(layers.Embedding(input_dim=self.n_chars + 2, output_dim=10,
                                    input_length=self.max_word_length))(char_in)

            char_enc = layers.TimeDistributed(layers.LSTM(units=self.char_hidden_dim, return_sequences=False))(emb_char)

            model_input = [word_in, char_in]
            x = layers.concatenate([emb_word, char_enc])
        else:
            model_input = word_in
            x = emb_word

        # Create input layer

        hidden_dim = layers.Bidirectional(layers.LSTM(units=self.hidden_dim, return_sequences=True))(x)
        model_output = layers.TimeDistributed(layers.Dense(n_classes, activation=self.activation_function))(hidden_dim)

        self.model = keras.Model(model_input, model_out)

    def compile_model(self, metrics=['acc'], loss='categorical_crossentropy', optimizer='adam'):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train_model(self, x, y, bach_size, epochs, val_split):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return

        print("[*] Starting training")
        self.history = self.model.fit(x, y, batch_size=bach_size, epochs=epochs, validation_split=val_split)

    def show_model(self):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        print("[*] Model structure")
        self.model.summary()

    def plot_model(self):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        print("[*] Model plot")
        keras.utils.plot_model(model, "lstm_model.png", show_shapes=True)

    def show_history(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc="lower right")
        plt.show()

    def evaluate(self, test_set):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        
        if not self.model.is_trained():
            print("[*] Error: The model has not been yet trained")
            return
  
    def predict(self, new_sentence):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        
        if not self.model.is_trained():
            print("[*] Error: The model has not been yet trained")
            return

        model.predict(new_sentence)