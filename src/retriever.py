"""
Retriever: given a sub-question and a corpus, returns top-K relevant passages.

Uses sentence-transformers for embeddings and FAISS for similarity search.
"""

from typing import List, Tuple
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class Passage(BaseModel):
    """A retrieved passage with metadata."""
    title: str
    text: str
    score: float = Field(description="Similarity score (higher = more relevant)")


class Retriever:
    """FAISS-based retriever over a corpus of passages."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.passages: List[dict] = []
    
    def build_index(self, passages: List[dict]):
        """
        Build FAISS index from passages.
        Each passage: {"title": str, "text": str}
        """
        self.passages = passages
        texts = [p["text"] for p in passages]
        
        print(f"Encoding {len(texts)} passages...")
        embeddings = self.embedder.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Index built: {self.index.ntotal} vectors, dim={dim}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        """Return top-k passages for the query."""
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieving.")
        
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            p = self.passages[idx]
            results.append(Passage(
                title=p["title"],
                text=p["text"],
                score=float(score)
            ))
        return results


def load_passages_from_hotpot(hotpot_example: dict) -> List[dict]:
    """
    Extract passages from a HotpotQA example.
    HotpotQA structure: context = {"title": [...], "sentences": [[...], ...]}
    """
    titles = hotpot_example["context"]["title"]
    sentences_per_title = hotpot_example["context"]["sentences"]
    
    passages = []
    for title, sents in zip(titles, sentences_per_title):
        text = " ".join(sents)
        passages.append({"title": title, "text": text})
    return passages


# ============ TEST ============
if __name__ == "__main__":
    import json
    
    # Load one HotpotQA example
    with open("data/hotpot_sample.json") as f:
        data = json.load(f)
    
    example = data[0]
    print(f"Question: {example['question']}")
    print(f"Gold answer: {example['answer']}\n")
    
    # Build retriever on this example's passages
    passages = load_passages_from_hotpot(example)
    print(f"Loaded {len(passages)} passages from context\n")
    
    retriever = Retriever()
    retriever.build_index(passages)
    
    # Retrieve for the question
    print(f"\nRetrieving top-3 for: '{example['question']}'\n")
    results = retriever.retrieve(example["question"], k=3)
    
    for i, r in enumerate(results, 1):
        print(f"--- Rank {i} (score={r.score:.3f}) ---")
        print(f"Title: {r.title}")
        print(f"Text: {r.text[:200]}...")
        print()