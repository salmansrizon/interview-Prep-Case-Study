"""Source package for the Zero-Shot Brand Content Generator.

Sub-packages, mirroring week 10's src/embeddings, src/extraction,
src/matching split — each concern gets its own module so app.py stays a
thin UI layer that only orchestrates calls into these:

    - src.llm         -> thin wrapper around the ``ollama`` package
    - src.prompting    -> zero-shot / few-shot / CoT prompt construction
    - src.security     -> prompt-injection delimiting + output validation
    - src.utils        -> shared helpers (logging)
"""
