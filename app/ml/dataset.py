import pickle
import re
import string
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from spacy.lang.pl import Polish

from . import params

RE_EMOJI = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)


class Dataset:
    def __init__(
        self,
        texts: Union[str, List[str]],
        tags: Optional[str] = None,
        clean_data: bool = True,
        remove_stopwords: bool = False,
        is_train: bool = True,
    ):
        self.clean_data = clean_data
        self.remove_stopwords = remove_stopwords
        self.is_train = is_train

        self.nlp = Polish()

        if self.is_train is True:
            self.df = self._build_dataframe_train(texts, tags)
        else:
            self.df = self._build_dataframe_infer(texts)

        self.word2idx, self.idx2word = self.build_dict()

    def _build_dataframe_train(self, texts_file, tags_file):
        with open(texts_file, encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            texts = pd.DataFrame(lines, columns=["text"])
        tags = pd.read_fwf(tags_file, header=None, names=["tag"])
        df = pd.concat([texts, tags], axis=1)
        df["tokens"] = df["text"].map(lambda x: self._preprocess_sentence(x))
        df["length"] = df["tokens"].map(lambda x: len(x))
        df["clean_text"] = df["tokens"].map(lambda x: " ".join(x))
        if self.clean_data:
            df = self.clean(df)
        return df

    def _build_dataframe_infer(self, lines: List[str]):

        df = pd.DataFrame(lines, columns=["text"])

        df["tokens"] = df["text"].map(lambda x: self._preprocess_sentence(x))
        df["length"] = df["tokens"].map(lambda x: len(x))
        df["clean_text"] = df["tokens"].map(lambda x: " ".join(x))
        if self.clean_data:
            df = self.clean(df)
        return df

    def _preprocess_sentence(self, sentence):
        sentence = (
            sentence.replace(r"\n", "")
            .replace(r"\r", "")
            .replace(r"\t", "")
            .replace("„", "")
            .replace("”", "")
        )
        doc = [tok for tok in self.nlp(sentence)]
        if not self.clean_data and str(doc[0]) == "RT":
            doc.pop(0)
        while str(doc[0]) == "@anonymized_account":
            doc.pop(0)
        while str(doc[-1]) == "@anonymized_account":
            doc.pop()
        if self.remove_stopwords:
            doc = [tok for tok in doc if not tok.is_stop]
        doc = [tok.lower_ for tok in doc]
        doc = [
            "".join(c for c in tok if not c.isdigit() and c not in string.punctuation)
            for tok in doc
        ]
        doc = [RE_EMOJI.sub(r"", tok) for tok in doc]
        doc = [tok.strip() for tok in doc if tok.strip()]
        return doc

    def build_dict(self):
        if self.is_train:
            sentences = self.df["tokens"]
            all_tokens = [token for sentence in sentences for token in sentence]
            words_counter = Counter(all_tokens).most_common()
            word2idx = {params.pad: 0, params.unk: 1}
            for word, _ in words_counter:
                word2idx[word] = len(word2idx)

            with open(params.word_dict_path, "wb") as dict_file:
                pickle.dump(word2idx, dict_file)

        else:
            with open(params.word_dict_path, "rb") as dict_file:
                word2idx = pickle.load(dict_file)

        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def print_stats(self):
        print(self.df["length"].describe())
        print(self.df["length"].quantile(0.95, interpolation="lower"))
        print(self.df["length"].quantile(0.99, interpolation="lower"))
        print(self.df.shape)
        print(self.df["tag"].value_counts())

    @staticmethod
    def get_random_emb(length):
        return np.random.uniform(-0.25, 0.25, length)

    @staticmethod
    def clean(dataframe):
        dataframe = dataframe.drop_duplicates("clean_text")
        return dataframe[
            (dataframe["tokens"].apply(lambda x: "rt" not in x[:1]))
            & (dataframe["length"] > 1)
        ]
