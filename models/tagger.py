import tensorflow as tf
import keras
from keras import layers

class SeqTagger:

    def __init__(self, n_cats, n_words, max_sent_length, max_word_length, hidden_dim, activation, char_embs=True, n_chars=None, char_hidden_dim=-1):
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
        self.activation_function = activation
        
        self.model = None
        self.history = None

    def build_model(self):
        ''' Build the neural tagger according to the specified config in the initialization'''
        
        # create input words layer
        word_in = layers.Input(
                        shape=(self.max_sent_length,), 
                        name="word_in")

        emb_word = layers.Embedding(
                        input_dim=self.n_words, 
                        output_dim=self.hidden_dim,
                        input_length=self.max_sent_length,
                        name="word_emb", 
                        mask_zero=True)(word_in)

        if self.char_embs:
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
                            layers.Bidirectional(
                                layers.LSTM(
                                    units=self.hidden_dim, 
                                    return_sequences=False), 
                                name="char_bilstm"))(emb_char)

            model_input = [word_in, char_in]
            x = layers.concatenate([emb_word, char_enc])
        
        else:
            model_input = word_in
            x = emb_word

        hidden_dim = layers.Bidirectional(
                            layers.LSTM(
                                units=self.hidden_dim, 
                                return_sequences=True),
                            name="bilstm")(x)

        model_output = layers.TimeDistributed(
                            layers.Dense(
                                units=self.n_cats, 
                                activation=self.activation_function), 
                            name="inference")(hidden_dim)

        print("*** MODEL")
        print("    max_sent_length = ",self.max_sent_length)
        print("    max_word_length = ",self.max_word_length)
        self.model = keras.Model(model_input, model_output)

    def compile_model(self, loss, optimizer, learning_rate):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        
        if optimizer=='adam':
            optim  = keras.optimizers.Adam(learning_rate)
        elif optimizer=='sgd':
            optim = keras.optimizers.SGD(learning_rate)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    
    def train_model(self, x, y, bach_size, epochs, val_split):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return

        print("[*] Starting training")
        self.history = self.model.fit(x, y, batch_size=bach_size, epochs=epochs, validation_split=val_split)
    
    def train_model(self, x, y, bach_size, epochs, xval, yval):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return

        print("[*] Starting training")
        self.history = self.model.fit(x, y, batch_size=bach_size, epochs=epochs, validation_data=(xval, yval))

    def show_model(self):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        print("[*] Model structure")
        self.model.summary()

    def show_history(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc="lower right")
        plt.show()

    def save_model(self, out_path):
        self.model.save(out_path+"/model.h5", overwrite=True)

    def evaluate(self, test_set):
        if self.model is None:
            print("[*] Errror: Model has not been yet created")
            return
        
        if not self.model.is_trained():
            print("[*] Error: The model has not been yet trained")
            return