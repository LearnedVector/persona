import numpy as np
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Embedding


def IntentModel(model):
    model = model.lower()
    if model == "onehot":
        return OneHotModel
    elif model == "embeddings":
        return EmbeddingsModel
    else:
        print("{} does not exist".format(model))
        return None


class BaseIntentModel:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.model = None
        self.history = None

    def train(self, optimizer='rmsprop', loss='categorical_crossentropy',
              batch_size=64, epochs=100, validation=0.0, summary=False):
        if summary:
            self.model.summary()

        self.model.compile(optimizer, loss)
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation)

    def decode(self, input_sequence, output_word_model):
        output_tokens = self.model.predict(input_sequence)
        token_index = np.argmax(output_tokens[0])
        intent = output_word_model.index2word[token_index]
        confidence = max(output_tokens[0])
        return intent, confidence


class OneHotModel(BaseIntentModel):
    def __init__(self, x_train, y_train,
                 input_len, output_len,
                 latent_dim=128, activation='relu'):
        self.x_train = x_train
        self.y_train = y_train
        self.model = \
            self.build_model(input_len, output_len, latent_dim, activation)

    def build_model(self, input_len, output_len, latent_dim, activation):
        model = Sequential()
        model.add(
            LSTM(latent_dim, activation=activation,
                 input_shape=(None, input_len)))
        model.add(Dense(output_len, activation='softmax'))
        return model


class EmbeddingsModel(BaseIntentModel):
    def __init__(self, x_train, y_train,
                 input_dim, output_dim,
                 latent_dim=128, activation='relu'):
        self.x_train = x_train
        self.y_train = y_train
        self.model = \
            self.build_model(input_dim, output_dim, latent_dim, activation)

    def build_model(self, input_dim, output_dim, latent_dim, activation):
        model = Sequential()
        model.add(Embedding(input_dim, output_dim,))
        model.add(
            LSTM(latent_dim, activation=activation,
                 input_shape=(None, input_dim)))
        model.add(Dense(output_dim, activation='softmax'))
        return model
