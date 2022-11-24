import keras
import pickle
import matplotlib.pyplot as plt
from keras import layers

class SeqTagger:
    '''
        Generic sequence labeling model. 
        Allows for word-level embeddings or character embeddings.
    '''

    def __init__(self, n_cats=17, n_words=1000, max_sent_length=128, max_word_length=16, hidden_dim=32, activation='adam', dropout=0.3, n_chars=100, char_hidden_dim=16):
        # Dataset settings
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.n_cats = n_cats
        self.n_words = n_words
        self.n_chars = n_chars
        
        # Obtain data from the tagger architecture
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        self.activation_function = activation
        self.dropout = dropout
        
        self.model = None
        self.history = None

    def build_model(self):
        ''' Build the neural tagger according to the specified config in the initialization'''
        
        # Create input words layer
        word_in = layers.Input(
                        shape=(self.max_sent_length,), 
                        name="word_in")

        emb_word = layers.Embedding(
                        input_dim=self.n_words, 
                        output_dim=self.hidden_dim,
                        input_length=self.max_sent_length,
                        name="word_emb", 
                        mask_zero=True)(word_in)

        # Character embeddings
        char_in = layers.Input(
                    shape=(self.max_sent_length, self.max_word_length,),
                    name="char_in")
        
        emb_char = layers.TimeDistributed(
                        layers.Embedding(
                            input_dim=self.n_chars, 
                            output_dim=self.char_hidden_dim, 
                            input_length=self.max_word_length, 
                            mask_zero=True), 
                        name="char_emb")(char_in)

        char_enc = layers.TimeDistributed(
                        layers.LSTM(
                                units=self.hidden_dim, 
                                return_sequences=False), 
                            name="char_lstm")(emb_char)

        model_input = [word_in, char_in]
        x = layers.concatenate([emb_word, char_enc])

        # Dropout
        x = layers.SpatialDropout1D(self.dropout)(x)        

        # Main lstms
        hidden_dim = layers.Bidirectional(
                            layers.LSTM(
                                units=self.hidden_dim, 
                                return_sequences=True),
                            name="bilstm")(x)

        # Inference layer
        model_output = layers.TimeDistributed(
                            layers.Dense(
                                units=self.n_cats, 
                                activation=self.activation_function), 
                            name="inference")(hidden_dim)

        print("*** VOCABULARY")
        print("    chars_vocab       =", self.n_chars)
        print("    words_vocab       =", self.n_words)
        print("    tags_vocab        =", self.n_cats)
        print("*** MODEL")
        print("    max_sent_length  =", self.max_sent_length)
        print("    hidden_dime      =", self.hidden_dim)
        print("    max_word_length  =", self.max_word_length)
        print("    char hidden_dim  =", self.char_hidden_dim)
        print("    dropout %        =", self.dropout)
        print("    activation_fun   =", self.activation_function)
        self.model = keras.Model(model_input, model_output)

    def compile_model(self, loss, optimizer, learning_rate, decay=False, lr_decay=-1, decay_steps=-1):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        
        if decay:
          lr = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=learning_rate,
          decay_steps=lr_decay,
          decay_rate=lr_decay)

        if optimizer=='adam':
            optim  = keras.optimizers.Adam(learning_rate)
        elif optimizer=='sgd':
            optim = keras.optimizers.SGD(learning_rate)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        print("*** COMPILATION")
        print("    optimizer        =", optimizer)
        print("    use_decay        =", decay)
        print("    lr_decay         =", lr_decay)
        print("    lr_decay_steps   =", decay_steps)
        print("    loss_fucntion    =", loss)
    
    def train_model(self, x, y, batch_size, epochs, val_split, early_stop=False, patience=-1):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return

        print("[*] Starting training")
        print("    batch_size       =", batch_size)
        print("    epochs           =", epochs)
        print("    early_stop       =", early_stop)
        print("    stop_patience    =", patience)
        
        self.history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=val_split)
    
    def train_model(self, x, y, batch_size, epochs, xval, yval, early_stop=False, patience=-1):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        if early_stop:
          early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)

        print("[*] Starting training")
        print("    batch_size       =", batch_size)
        print("    epochs           =", epochs)
        print("    early_stop       =", early_stop)
        print("    stop_patience    =", patience)
        
        self.history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(xval, yval))

    def load_model(self, model_path):
        if self.model is not None:
            print("[*] Model already loaded")
            return
        else:
            self.model = keras.models.load_model(model_path+"/model.h5")
            with open(model_path+'/model.history', "rb") as file_pi:
                self.model.history = pickle.load(file_pi)
            print("[*] Model loaded from", model_path)
    
    def show_model(self):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        print("[*] Model structure")
        self.model.summary()

    def show_history(self):
        print(self.history)

    def plot_history(self, save_file=False, filename="history.png"):
        h = self.model.history

        fig, (fig_1, fig_2) = plt.subplots(2, figsize=(15, 15))

        fig_1.set_title('Accuracy')
        fig_1.plot(h['acc'], color='blue', label='Training')
        fig_1.plot(h['val_acc'], color='red', label='Validation')
        fig_1.set_ylim([0, 1])

        x_tr = len(h['acc'])-1
        y_tr = h['acc'][-1]
        text_tr = "{:.2f}".format(100*y_tr)+"%"

        fig_1.annotate(text_tr,xy=(x_tr,y_tr))

        x_val = len(h['val_acc'])-1
        y_val = h['val_acc'][-1]
        text_val = "{:.2f}".format(100*y_val)+"%"

        fig_1.annotate(text_val,xy=(x_val,y_val))

        fig_2.set_title('Loss')
        fig_2.plot(h['loss'], color='blue', label='Training')
        fig_2.plot(h['val_loss'], color='red', label='Validation')
        fig_2.set_ylim([0, 0.5])

        x_tr = len(h['loss'])-1
        y_tr = h['loss'][-1]
        text_tr = "{:.2f}".format(100*y_tr)+"%"

        fig_2.annotate(text_tr,xy=(x_tr,y_tr))

        x_val = len(h['val_loss'])-1
        y_val = h['val_loss'][-1]
        text_val = "{:.2f}".format(100*y_val)+"%"

        fig_2.annotate(text_val,xy=(x_val,y_val))


        fig.legend(loc='lower right')
        fig.show()

        if save_file:
          plt.savefig(filename)

    def save_model(self, out_path):
        self.model.save(out_path+"/model.h5", overwrite=True)
        with open(out_path+"/model.history", 'wb') as file_pi:
            pickle.dump(self.model.history.history, file_pi)

    def evaluate(self, test_set):
        if self.model is None:
            print("[*] Error: Model has not been yet created")
            return

        x_test, y_test = test_set
        results = self.model.evaluate(x_test, y_test)
        print("[*] Test loss || Test acc:", results)

    def decode(self, x):
      return self.model.predict(x)
