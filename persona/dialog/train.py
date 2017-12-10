#!/usr/bin/env python3

import sys
sys.path += ['.']  # qa

from persona.preprocess import prepare_json_data, pad_sequences, one_hot_encode, one_hot_encode_target

MAX_SEQ_LEN = 15
input_word_model, output_word_model, pairs = \
    prepare_json_data('input', 'output', './data/persona.dialog.json', MAX_SEQ_LEN)

input_seqs = [pair[0] for pair in pairs]
output_seqs = [pair[1] for pair in pairs]

padded_input = pad_sequences(input_seqs, MAX_SEQ_LEN)
padded_output = pad_sequences(output_seqs, MAX_SEQ_LEN)

one_hot_input = one_hot_encode(padded_input, input_word_model, MAX_SEQ_LEN)
one_hot_output = one_hot_encode(padded_output, output_word_model, MAX_SEQ_LEN)

one_hot_target = one_hot_encode_target(padded_output, output_word_model, MAX_SEQ_LEN)

from persona.model.dialog import DialogModel

model = DialogModel(one_hot_input, one_hot_output, one_hot_target)
model.encoder(input_word_model.n_words)
model.decoder(output_word_model.n_words)
model.train(optimizer='adam', epochs=100, summary=True)

from persona.preprocess import pad_sequence

try:
    while True:
        _input = input("input: ")
        padded_input = [pad_sequence(_input, MAX_SEQ_LEN).split()]
        one_hot = one_hot_encode(padded_input, input_word_model, MAX_SEQ_LEN)
        prediction, confidence = model.decode(one_hot, output_word_model, MAX_SEQ_LEN)
        print("response: ", prediction, confidence)
except KeyboardInterrupt:
    pass


