
import re
from pathlib import Path

import chromadb
import ollama
from langchain.tools import tool
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


CHROMA_PATH = Path(__file__).resolve().parent.parent / "chromadb"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
MIN_RERANK_SCORE = None  

col = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)



def embed_query(query: str) -> list[float]:
    """Embed a question using the SAME call as ingestion."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]


def vector_search(query: str, k: int = 10, where: dict | None = None) -> list[dict]:
    """Semantic search in Chroma. Returns chunk dicts, best match first."""
    res = col.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for id_, text, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append({
            "id": id_,
            "text": text,
            "page": meta.get("page"),
            "heading": meta.get("heading"),
            "source": meta.get("source"),
            "distance": dist,
        })
    return chunks



_bm25 = None
_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Lowercase words only. Corpus and query MUST use this same function."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index() -> None:
    """(Re)build the keyword index from everything stored in Chroma.
    Call again after ingesting another paper."""
    global _bm25, _corpus
    data = col.get(include=["documents", "metadatas"])
    _corpus = [
        {
            "id": id_,
            "text": text,
            "page": meta.get("page"),
            "heading": meta.get("heading"),
            "source": meta.get("source"),
        }
        for id_, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _corpus])


build_bm25_index()
if not _corpus:
    raise RuntimeError(f"Chroma collection is empty. Check CHROMA_PATH: {CHROMA_PATH}")

def bm25_search(query: str, k: int = 10, source: str | None = None) -> list[dict]:
    """Keyword search. Same dict shape as vector_search. Filters by source in Python."""
    scores = _bm25.get_scores(_tokenize(query))
    hits = []
    for idx in scores.argsort()[::-1]:
        if scores[idx] <= 0:
            break  
        chunk = _corpus[idx]
        if source and chunk["source"] != source:
            continue
        hits.append({**chunk, "bm25": float(scores[idx])})
        if len(hits) == k:
            break
    return hits



def merge(vector_hits: list[dict], keyword_hits: list[dict], k: int = 10, rrf_k: int = 60) -> list[dict]:
    """Combine two ranked lists by RANK position (their raw scores are not comparable)."""
    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, hit in enumerate(hits):
            by_id.setdefault(hit["id"], hit)
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
    best_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**by_id[i], "rrf": round(fused[i], 5)} for i in best_ids]



_reranker = None


def _get_reranker() -> CrossEncoder:
    """Load lazily: the model is large, so only pay for it when first used."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, max_length=256) 
    return _reranker


def rerank(query: str, chunks: list[dict], top_n: int = 4) -> list[dict]:
    """Score each (question, chunk) pair together; keep the best top_n."""
    if not chunks:
        return []
    scores = _get_reranker().predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [{**chunk, "rerank": float(score)} for chunk, score in ranked]



def retrieve_chunks(
    query: str,
    source: str | None = None,
    k_candidates: int = 10,
    top_n: int = 4,
    min_score: float | MIN_RERANK_SCORE = 0.2 ,
) -> list[dict]:
    """Full retrieval. `source` limits search to one paper (used by compare workers).
    Returns [] when nothing clears min_score, so the agent can say 'not found'."""
    where = {"source": source} if source else None
    vec_hits = vector_search(query, k=k_candidates, where=where)
    kw_hits = bm25_search(query, k=k_candidates, source=source)
    candidates = merge(vec_hits, kw_hits, k=k_candidates)
    top = rerank(query, candidates, top_n=top_n)
    if min_score is not None:
        top = [c for c in top if c["rerank"] >= min_score]
    return top



def format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant passages found in the papers."
    return "\n\n".join(
        f"[{c['source']} | p.{c['page']} | {c['heading']}]\n{c['text']}" for c in chunks
    )


@tool
def retrieve(query: str) -> str:
    """Search the research papers for passages relevant to the query.
    Use for any question about a paper's content, methods, datasets or results.
    Returns passages labelled with source, page and section heading."""
    return format_chunks(retrieve_chunks(query))




  
