import os
import pickle
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from spacy.lang.pl import Polish

WORD_DICT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/word_dict.pickle"
)

amoniminimized_accounts_xddd = (
    "@anonymized_account",
    "@anonifikowane_account",
    "@anonifikowany_account",
    "@anonimizized_account",
    "@anonimized_account",
    "@anononymized_account",
    "@anononized_account",
    "@anonimized_aconimount",
)


class Dataset:
    """Container for dataset; this class is required to run the model."""

    def __init__(
        self,
        texts: Union[List[str], str],
        tags: Optional[str] = None,
        clean_data: bool = False,
        remove_stopwords: bool = False,
        is_train: bool = False,
    ):
        """
        Initialises the Dataset object.

        Depending on mode (training or inference) it stores and preprocess corpus and
        tags attached to each sentence.

        :param texts:
        :param tags: path to text file with tags for dataset; use only if training
        :param clean_data: a flag - if true data is cleaned (i.e. small sentences are
            removed; use only if training
        :param remove_stopwords: a flag - if true stopwords are removed from tokenized
            sentence; use only if training
        :param is_train: a flag that indicates if dataset is created for training or to
            inference the model
        """
        self.clean_data = clean_data
        self.remove_stopwords = remove_stopwords
        self.is_train = is_train

        self.nlp = Polish()

        if self.is_train is True:
            _df = self._text_tag_files_to_df(texts, tags)  # type:ignore
        else:
            _df = pd.DataFrame(texts, columns=["text"])

        self.df = self._build_dataframe(_df)
        self.word2idx, self.idx2word = self._build_dict()

    def _build_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add to corpus dataframe parameters for model inference."""
        df["tokens"] = df["text"].map(lambda x: self._preprocess_sentence(x))
        df["length"] = df["tokens"].map(lambda x: len(x))
        df["clean_text"] = df["tokens"].map(lambda x: " ".join(x))
        if self.clean_data:
            df = self._clean(df)
        return df

    def _preprocess_sentence(self, sentence: str) -> List[str]:
        """Tokenize sentence and clears unnecessary tokens."""
        re_emoji = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        sentence = sentence.lower()
        amoniminimized_account_correct = "@anonymized_account"
        sentence = (
            sentence.replace(r"\n", "")
            .replace(r"\r", "")
            .replace(r"\t", "")
            .replace("„", "")
            .replace("”", "")
            .replace("@anonymized_account", amoniminimized_account_correct)
            .replace("@anonifikowane_account", amoniminimized_account_correct)
            .replace("@anonifikowanym_account", amoniminimized_account_correct)
            .replace("@anonifikowany_account", amoniminimized_account_correct)
            .replace("@anonimizized_account", amoniminimized_account_correct)
            .replace("@anonimized_account", amoniminimized_account_correct)
            .replace("@anononymized_account", amoniminimized_account_correct)
            .replace("@anononized_account", amoniminimized_account_correct)
            .replace("@anonimized_aconimount", amoniminimized_account_correct)
        )
        doc = [tok for tok in self.nlp(sentence)]
        if not self.clean_data and str(doc[0]) == "RT":
            doc.pop(0)
        while str(doc[0]) == amoniminimized_account_correct:
            doc.pop(0)
        while str(doc[-1]) == amoniminimized_account_correct:
            doc.pop()
        if self.remove_stopwords:
            doc = [tok for tok in doc if not tok.is_stop]
        doc = [tok.lower_ for tok in doc]
        doc = [
            "".join(c for c in tok if not c.isdigit() and c not in string.punctuation)
            for tok in doc
        ]
        doc = [re_emoji.sub(r"", tok) for tok in doc]
        doc = [tok.strip() for tok in doc if tok.strip()]
        return doc

    def _build_dict(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Reads mappings of words to indices."""
        if self.is_train:
            sentences = self.df["tokens"]
            all_tokens = [token for sentence in sentences for token in sentence]
            words_counter = Counter(all_tokens).most_common()
            word2idx = {"<PAD>": 0, "<UNK>": 1}
            for word, _ in words_counter:
                word2idx[word] = len(word2idx)

            with open(WORD_DICT_PATH, "wb") as dict_file:
                pickle.dump(word2idx, dict_file)

        else:
            with open(WORD_DICT_PATH, "rb") as dict_file:
                word2idx = pickle.load(dict_file)

        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def print_stats(self) -> None:
        """Print statistics about model."""
        print(self.df["length"].describe())
        print(self.df["length"].quantile(0.95, interpolation="lower"))
        print(self.df["length"].quantile(0.99, interpolation="lower"))
        print(self.df.shape)
        if self.is_train:
            print(self.df["tag"].value_counts())

    @staticmethod
    def _text_tag_files_to_df(texts_file: str, tags_file: str) -> pd.DataFrame:
        """Read out corpus and tags from paths and concatenate to dataframe."""
        with open(texts_file, encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            texts = pd.DataFrame(lines, columns=["text"])
        tags = pd.read_fwf(tags_file, header=None, names=["tag"])
        return pd.concat([texts, tags], axis=1)

    @staticmethod
    def _clean(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Clean the data from one token tweets and retweets."""
        dataframe = dataframe.drop_duplicates("clean_text")
        return dataframe[
            (dataframe["tokens"].apply(lambda x: "rt" not in x[:1]))
            & (dataframe["length"] > 1)
        ]
