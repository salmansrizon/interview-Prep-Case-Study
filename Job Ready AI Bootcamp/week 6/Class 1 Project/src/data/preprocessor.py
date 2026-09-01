"""
Text preprocessing pipeline.
Demonstrates tokenization, stopword removal, and lemmatization —
core concepts before vectorization.
"""

import re
import string
from typing import List

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextPreprocessor:
    """
    Production-grade text cleaner for SMS/email-style text.
    """

    def __init__(self, language: str = "english") -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            # Keep the app fully offline when optional NLTK corpora are absent.
            self.stop_words = set(ENGLISH_STOP_WORDS)
        # Keep negation words as they flip sentiment/intent
        self.stop_words -= {"no", "not", "nor", "neither", "never"}

    def clean(self, text: str) -> str:
        """Run full preprocessing pipeline on a single document."""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove numbers (optional: keep if intent relies on them)
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        try:
            tokens: List[str] = word_tokenize(text)
        except LookupError:
            # Regex tokenization is sufficient after punctuation removal and
            # avoids downloading NLTK's optional punkt data at runtime.
            tokens = re.findall(r"[a-z]+", text)

        # Remove stopwords & lemmatize
        normalized_tokens = []
        for token in tokens:
            if token in self.stop_words or len(token) <= 2:
                continue
            try:
                normalized_tokens.append(self.lemmatizer.lemmatize(token))
            except LookupError:
                # PorterStemmer ships with NLTK and needs no downloaded corpus.
                normalized_tokens.append(self.stemmer.stem(token))

        return " ".join(normalized_tokens)

    def transform_series(self, texts: List[str]) -> List[str]:
        """Batch process a list of texts."""
        logger.info("Preprocessing %d documents...", len(texts))
        return [self.clean(t) for t in texts]
