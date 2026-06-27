import faiss
import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, embeddings, chunks):
        self.chunks = chunks
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))

        # BM25 setup
        self.tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def retrieve_vector(self, query_emb, k=3):
        query_emb = np.asarray(query_emb)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        D, I = self.index.search(query_emb, k)
        return [self.chunks[i] for i in I[0]]

    def retrieve_bm25(self, query, k=3):
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        # Top k by BM25
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in bm25_top_indices]

    def retrieve_hybrid(self, query, model, k_bm25=3, k_vector=3):
        query_emb = model.encode([query])
        vector_chunks = self.retrieve_vector(query_emb, k=k_vector)
        bm25_chunks = self.retrieve_bm25(query, k=k_bm25)

        # Merge and deduplicate, BM25 results first
        seen = set()
        hybrid_chunks = []
        for chunk in bm25_chunks + vector_chunks:
            if chunk not in seen:
                hybrid_chunks.append(chunk)
                seen.add(chunk)
        return hybrid_chunks
