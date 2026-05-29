"""
Sanity checks for the preprocessing pipeline.
Run with: pytest tests/test_pipeline.py
"""

from src.data.preprocessor import TextPreprocessor


def test_preprocessor_lowercase():
    p = TextPreprocessor()
    assert p.clean("HELLO WORLD") == "hello world"


def test_preprocessor_removes_urls():
    p = TextPreprocessor()
    result = p.clean("Check out https://example.com for more info")
    assert "http" not in result
    assert "example" in result  # domain name without protocol remains


def test_preprocessor_removes_punctuation():
    p = TextPreprocessor()
    result = p.clean("Hello, world!!!")
    assert "," not in result
    assert "!" not in result


def test_preprocessor_lemmatization():
    p = TextPreprocessor()
    result = p.clean("running runs runner")
    assert "run" in result
