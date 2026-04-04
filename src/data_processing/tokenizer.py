import os
import pickle
from typing import List

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextTokenizer:
    def __init__(self, num_words: int = 10000, oov_token: str = "<OOV>", max_length: int = 200):
        self.num_words = num_words
        self.oov_token = oov_token
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    def fit(self, texts: List[str]):
        """
        Fit tokenizer on training texts
        """
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts: List[str]):
        """
        Convert texts to padded sequences
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        return padded_sequences

    def fit_transform(self, texts: List[str]):
        """
        Fit and transform texts
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        """
        Save tokenizer to disk
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                "tokenizer": self.tokenizer,
                "max_length": self.max_length,
                "num_words": self.num_words,
                "oov_token": self.oov_token
            }, f)

    @staticmethod
    def load(path: str):
        """
        Load tokenizer from disk
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        obj = TextTokenizer(
            num_words=data["num_words"],
            oov_token=data["oov_token"],
            max_length=data["max_length"]
        )
        obj.tokenizer = data["tokenizer"]
        return obj