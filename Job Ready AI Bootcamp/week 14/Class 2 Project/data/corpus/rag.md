# Retrieval-Augmented Generation (RAG)

## The idea
RAG answers a question in two steps. Retrieval finds the most relevant chunks from a document store, and generation asks the language model to answer using only those chunks. Grounding the answer in retrieved text reduces hallucination and allows citations.

## Pipeline
A typical pipeline extracts text from documents, splits it into chunks, embeds the chunks, and stores them in a vector database. At question time the question is embedded, the top-K chunks are retrieved (K between 3 and 5 is a common choice), and an augmented prompt containing those chunks is sent to the model.

## Guardrails
A vector search always returns its nearest neighbours, even for an unrelated question, because it has no concept of "nothing matches". A distance threshold can skip the language model call entirely when the best chunk is too far away, and the system prompt can tell the model to say "not found" instead of guessing.

## Citations
Storing the source file and page with every chunk lets the application show where each answer came from. Citations let a reader check a claim instead of trusting it.
