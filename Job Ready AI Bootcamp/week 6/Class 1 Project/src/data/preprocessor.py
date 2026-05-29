"""
Text preprocessing pipeline.
Demonstrates tokenization, stopword removal, and lemmatization —
core concepts before vectorization.
"""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure NLTK data is available (runs offline if already downloaded)
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


class TextPreprocessor:
    """
    Production-grade text cleaner for SMS/email-style text.
    """

    def __init__(self, language: str = "english") -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
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
        tokens: List[str] = word_tokenize(text)

        # Remove stopwords & lemmatize
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 2
        ]

        return " ".join(tokens)

    def transform_series(self, texts: List[str]) -> List[str]:
        """Batch process a list of texts."""
        logger.info("Preprocessing %d documents...", len(texts))
        return [self.clean(t) for t in texts]
