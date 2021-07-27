import os

_lib_path = os.path.dirname(os.path.abspath(__file__))

pad = "<PAD>"
unk = "<UNK>"
emb_size = 300
word_dict_path = os.path.join(_lib_path, "data/word_dict.pickle")
max_sent_length = 20
