import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense


class DialogModel:
    def __init__(self, encoder_input_data,
                 decoder_input_data, decoder_target_data):
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data

        # intialize attributes for encoder, decoder model
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_outputs = None
        self.decoder_latent_dim = None
        self.history = None

        # initializing attributes for inference model
        self.encoder_model = None
        self.decoder_model = None
        self._decoder = None
        self._decoder_dense = None
        self.inference_ready = False

    def encoder(self, input_len, latent_dim=128, activation='relu'):
        self.encoder_inputs = Input(shape=(None, input_len))
        encoder = LSTM(latent_dim, activation=activation, return_state=True)
        _, state_h, state_c = encoder(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]

    def decoder(self, output_len, latent_dim=128, activation='relu'):
        self.decoder_latent_dim = latent_dim
        self.decoder_inputs = Input(shape=(None, output_len))
        self._decoder = LSTM(
            self.decoder_latent_dim, activation=activation,
            return_sequences=True, return_state=True)
        decoder_outputs, _, _ = \
            self._decoder(self.decoder_inputs,
                          initial_state=self.encoder_states)
        self._decoder_dense = Dense(output_len, activation='softmax')
        self.decoder_outputs = self._decoder_dense(decoder_outputs)

    def train(self, optimizer='rmsprop', loss='categorical_crossentropy',
              batch_size=64, epochs=100, validation=0.0, summary=False):
        model = Model(
            [self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

        if summary:
            model.summary()

        model.compile(optimizer, loss)
        self.history = model.fit(
            [self.encoder_input_data, self.decoder_input_data],
            self.decoder_target_data, batch_size=batch_size,
            epochs=epochs, validation_split=validation)

    def build_inference_model(self):
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        latent_dim = self.decoder_latent_dim
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self._decoder(
            self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self._decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        self.inference_ready = True

    def decode(self, input_sequence, output_word_model, max_sentence_len):
        if not self.inference_ready:
            self.build_inference_model()

        state_values = self.encoder_model.predict(input_sequence)
        output_len = output_word_model.n_words
        target_seq = np.zeros((1, 1, output_len))
        target_seq[0, 0, output_word_model.word2index['SOS']] = 1
        stop_cond = False
        decoded_sent = []
        confidence = 0
        while not stop_cond:
            output_tokens, h, c = \
                self.decoder_model.predict([target_seq] + state_values)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            word = output_word_model.index2word[sampled_token_index]
            if word == 'EOS' or len(decoded_sent) > max_sentence_len:
                stop_cond = True
            else:
                confidence += max(output_tokens[0, -1, :])
                decoded_sent.append(word)
            target_seq = np.zeros((1, 1, output_len))
            target_seq[0, 0, output_word_model.word2index[word]] = 1
            state_values = [h, c]
        return " ".join(decoded_sent), confidence/len(decoded_sent)
