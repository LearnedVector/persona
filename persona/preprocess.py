import numpy as np
from io import open
import unicodedata
import string
import re
import json

PAD = "PAD"
SOS = "SOS"
EOS = "EOS"
UNK = "UNK"


class WordModel:
    def __init__(self, name):
        self.name = name
        self.word2index = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.word2count = {}
        self.index2word = {0: PAD, 1: SOS, 2: EOS, 3: UNK}
        self.n_words = 4  # count for default tokens

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# helper functions to prepocess
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s.lower().rstrip())
    s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.=!?{}:]+", r" ", s)
    if s[-1] == ".":
        s = s[:-1]+" ."
    return s


def filter_pair(p):
    """" filter sequence by MAX_LENGTH """
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """ wrapper that filter sequence pairs by MAX_LENGTH """
    return [pair for pair in pairs if filter_pair(pair)]


def get_data_pairs_from_json(_input, _output, filename):
    """ """
    with open(filename, 'r') as f:
        data = json.load(f)
    pairs = []
    for pair in data:
        pairs.append(
            [normalize_string(pair[_input]), normalize_string(pair[_output])])
    input_word_model = WordModel(_input)
    output_word_model = WordModel(_output)
    return input_word_model, output_word_model, pairs


def prepare_json_data(_input, _output, filename, max_length=15):
    """ prepares data for processing. Expect json structure to follow...

            {
                "input": "input string"
                "output": "output string"
            }

        Args:
            _input (str): string of the key for model input
            _output (str): string of the key for model output
            filename (str): string of absolute path of file
            max_length (int): Maximum length of the sequence

        Returns:
            input_word_model (WordModel): object containing input defintiions
            output_word_model (WordModel): object containing output definitions
            pairs (list<str>): [input, output] sentences
    """
    global MAX_LENGTH
    MAX_LENGTH = max_length

    input_word_model, output_word_model, pairs = \
        get_data_pairs_from_json(_input, _output, filename)
    print("READ %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_word_model.add_sentence(pair[0])
        output_word_model.add_sentence(pair[1])
    print("Counted Words:")
    print(input_word_model.name, input_word_model.n_words)
    print(output_word_model.name, output_word_model.n_words)
    return input_word_model, output_word_model, pairs


def pad_sequence(sequence, seq_len):
    pad_len = (seq_len - 2) - len(sequence.split())
    pads = [PAD for i in range(pad_len)]
    _seq = " ".join([SOS, *sequence.split()[:seq_len-2], EOS, *pads])
    assert len(_seq.split()) == seq_len
    return _seq


def pad_sequences(sequences, seq_len):
    """ adds PAD, SOS, EOS, UNK to sequence """
    padded_seqs = []
    for seq in sequences:
        _seq = pad_sequence(seq, seq_len)
        padded_seqs.append(_seq.split())
    return padded_seqs


def integer_encode(sequences, word_model, seq_len):
    """ returns nparray index encoded sequences """
    encoded_list = np.zeros((len(sequences), seq_len))
    for i, seq in enumerate(sequences):
        for j, word in enumerate(seq):
            encoded_list[i, j] = word_model.word2index[word]
    return encoded_list


def one_hot_encode(sequences, word_model, seq_len):
    """ returns nparray with one hot encoded sequences """
    one_hot = np.zeros((len(sequences), seq_len, word_model.n_words))
    for i, seq in enumerate(sequences):
        for j, word in enumerate(seq):
            if word != PAD:
                if word not in word_model.word2index:
                    word = UNK
                one_hot[i, j, word_model.word2index[word]] = 1
    return one_hot


def one_hot_encode_target(sequences, word_model, seq_len):
    """" encode target with an offset of 1 for decoder in seq2seq """
    one_hot = np.zeros((len(sequences), seq_len, word_model.n_words))
    for i, seq in enumerate(sequences):
        for j, word in enumerate(seq):
            if word != PAD and j > 0:
                one_hot[i, j-1, word_model.word2index[word]] = 1
    return one_hot
