# Vector Databases

## What they do
A vector database stores embeddings (lists of numbers that capture meaning) and finds the vectors closest to a query vector. Comparing a query against every stored vector (brute force) becomes slow when there are millions of chunks, so vector databases use Approximate Nearest Neighbor (ANN) indexes that return close-enough neighbours in milliseconds.

## ChromaDB
ChromaDB is a local, open-source vector database that runs inside Python without a server. A PersistentClient stores collections on disk. Each chunk can carry metadata such as the source file and page number, which later powers citations and metadata filters.

## Semantic versus keyword search
Keyword search only finds exact words, so a query for "sick leave" misses a chunk that says "medical absence". Semantic search compares meaning, so both are found. Semantic search is weaker for exact identifiers such as "Policy #4471"; production systems often combine both in hybrid search.

## Chunking
Large documents are split into chunks before embedding. Recursive splitting tries paragraph breaks first, then sentences, so chunks do not cut sentences in half. A small overlap between neighbouring chunks keeps facts that sit on a chunk border. A practical range is about 300 to 800 characters with roughly 10 to 15 percent overlap; very small chunks lose context and very large chunks dilute the signal.

## Alternatives
FAISS is very fast and supports GPUs but does not manage metadata or persistence by itself. Pinecone is a managed cloud service, so data leaves the machine. Weaviate offers built-in hybrid search and can be self-hosted.
