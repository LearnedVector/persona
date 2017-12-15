from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate


def DialogModel(lstm_size, input_vocab_size, output_vocab_size, num_intents, num_dialogs, activation='relu'):
    """
    A dialog model used to predict conversational responses to natural language

    Args:
        lstm_size (int): number of recurrent units
        input_vocab_size (int): input vocabulary size
        output_vocab_size (int): output vocabulary size
        num_intents (int): number of intent classes the network can predict
        activation (str): type of activation function to use for LSTMs

    Returns:
        intent_model, dialog_model, combined_model:
    """
    sentence_input = Input(shape=(None, input_vocab_size))
    encoder = LSTM(lstm_size, activation=activation, return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder(sentence_input)
    encoder_states = [state_h, state_c]

    intent_engine = Dense(num_intents, activation='softmax')
    intent_output = intent_engine(Concatenate()(encoder_states))

    dialog_input = Input(shape=(num_dialogs,))
    influence_input = Concatenate(axis=-1)([state_c, dialog_input])
    influence_dense = Dense(lstm_size)
    altered_states = influence_dense(influence_input)

    decoder_lstm = LSTM(lstm_size, activation=activation, return_sequences=True)
    decoder_outputs = decoder_lstm(encoder_output, initial_state=[state_h, altered_states])

    decoder_dense = Dense(output_vocab_size, activation='softmax')
    response_output = decoder_dense(decoder_outputs)

    return (Model(sentence_input, intent_output),
            Model([sentence_input, dialog_input], response_output),
            Model([sentence_input, dialog_input], [intent_output, response_output]))
