# ChessMate — Hybrid RAG Chatbot

A domain-specific RAG chatbot answering chess questions 
by retrieving content from "Chess For Dummies" PDF using 
a hybrid FAISS + BM25 dual-retrieval pipeline.

## My Contribution
- Developed embedding generation using SentenceTransformer
- Built HybridRetriever class integrating FAISS vector 
  search and BM25 keyword search with deduplication

## Tech Stack
Python | FAISS | BM25 | SentenceTransformer | 
HuggingFace Transformers | pdfplumber | NLTK

## Evaluation
Precision, recall, and latency metrics across 
keyword and semantic queries
