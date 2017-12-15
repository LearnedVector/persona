#!/usr/bin/env python3
import json
import sys

from os.path import isfile
from traceback import print_exc

from keras.models import load_model

sys.path += ['.']  # noqa

import numpy as np
from persona.preprocess import probs_to_sent, normalize_string
from persona.dialog.model import DialogModel


def tokenize(s):
    return normalize_string(s).split(' ')


def vectorize_words(words, dictionary):
    one_hot = np.zeros((len(words), len(dictionary)))
    for i, word in enumerate(words):
        one_hot[i, dictionary.get(word, dictionary.get('UNK'))] = 1
    return one_hot


def gen_word2index(getter, extra_vocab=None, tokenizer=tokenize):
    words = sum([tokenizer(getter(i)) for i in data], [])
    return {w: i for i, w in enumerate(set(words + (extra_vocab or [])))}


def vectorize_in(sent):
    sent = list(reversed(tokenize(sent)))
    words = ['PAD'] * (MAX_SEQ_LEN - len(sent) - 1) + ['SOS'] + sent
    return vectorize_words(words, question_words)


def vectorize_out(sent):
    sent = tokenize(sent)
    words = sent + ['EOS'] * (MAX_SEQ_LEN - len(sent))
    return vectorize_words(words, response_words)


def vectorize_name(name, dictionary):
    vec = np.zeros(len(dictionary))
    vec[dictionary[name]] = 1
    return vec


MAX_SEQ_LEN = 15

with open('./data/persona.json') as f:
    data = json.load(f)

data = list(filter(lambda x: len(tokenize(x['input']['sentence'])) < MAX_SEQ_LEN and len(tokenize(x['output']['sentence'])) < MAX_SEQ_LEN, data))

question_words = gen_word2index(lambda x: x['input']['sentence'], ['PAD', 'SOS', 'UNK'])
response_words = gen_word2index(lambda x: x['output']['sentence'], ['EOS'])
response_id_to_word = {v: k for k, v in response_words.items()}

dialog_names = gen_word2index(lambda x: x['input']['dialog'], [], lambda x: [x])
intent_names = gen_word2index(lambda x: x['output']['intent'], [], lambda x: [x])
intent_id_to_name = {v: k for k, v in intent_names.items()}

question_data = np.array([vectorize_in(i['input']['sentence']) for i in data])
response_data = np.array([vectorize_out(i['output']['sentence']) for i in data])

dialog_data = np.array([vectorize_name(i['input']['dialog'], dialog_names) for i in data])
intent_data = np.array([vectorize_name(i['output']['intent'], intent_names) for i in data])

if False and isfile('intent.net') and isfile('dialog.net') and isfile('combined.net'):
    intent, dialog, combined = load_model('intent.net'), load_model('dialog.net'), load_model('combined.net')
else:
    intent, dialog, combined = DialogModel(200, len(question_words), len(response_words), len(intent_names), len(dialog_names))
    combined.compile('adam', 'categorical_crossentropy')
    combined.summary()

    try:
        combined.fit([question_data, dialog_data], [intent_data, response_data], epochs=200000)
    except KeyboardInterrupt:
        pass

    intent.save('intent.net')
    dialog.save('dialog.net')
    combined.save('combined.net')


def add_dim(v):
    return v.reshape((1, *v.shape))


while True:
    try:
        _input = input("> ")
        question_input = vectorize_in(_input)
        intent_confidences = intent.predict(add_dim(question_input))[0]
        intent_id = np.argmax(intent_confidences)
        intent_name = intent_id_to_name[intent_id]
        intent_conf = intent_confidences[intent_id]



        dialog_name = intent_name

        dialog_input = vectorize_name(dialog_name, dialog_names)
        dialog_output = dialog.predict([add_dim(question_input), add_dim(dialog_input)])

        response, response_conf = probs_to_sent(dialog_output[0], response_id_to_word)

        print('    ' + intent_name + ':', str(round(intent_conf, 2)) + ',', round(response_conf, 2))
        print('        ', response)
    except KeyboardInterrupt:
        raise
    except EOFError:
        raise
    except:
        print_exc()


