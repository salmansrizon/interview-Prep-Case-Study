# 🔎 Document Indexer & Semantic Search Demo

## সহজ ভাষায় Project Overview

এই notebook document-কে ছোট **chunk**-এ ভাগ করে embedding তৈরি করে ChromaDB-তে index করে। তারপর keyword match-এর বদলে meaning অনুযায়ী relevant text retrieve করে। এটি পরের RAG project-এর retrieval foundation।

## কীভাবে কাজ করে?

```text
Document → Chunking + Overlap → Embedding → ChromaDB
         → User Query → Similarity Search → Relevant Chunks
```

## কী শিখবেন?

- Fixed-size বনাম recursive/paragraph-aware chunking
- Chunk overlap কেন context preserve করে
- Keyword এবং Semantic Search-এর difference
- Persistent ChromaDB collection create ও query করা

## কীভাবে Run করবেন?

```powershell
cd "week 12\Class 1 Project"
python -m pip install jupyter chromadb sentence-transformers
python -m jupyter notebook document_indexer_demo.ipynb
```

প্রথম model download-এর জন্য internet লাগতে পারে; পরে cached model দিয়ে local execution সম্ভব।

## Validation Checklist

- Empty chunk index করা হয় না
- Chunk ID unique
- Query relevant section return করে
- Different chunk strategy-এর retrieval result compare করা হয়
- Re-running notebook duplicate/corrupt collection তৈরি করে না
