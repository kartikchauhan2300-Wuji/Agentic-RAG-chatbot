
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
MIN_RERANK_SCORE = None  # set AFTER calibrating with check 3f; None = never refuse

col = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)


# ---------- step 1-2: embedding + vector search (already tested) ----------
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
            "distance": dist,  # cosine distance: LOWER = closer match
        })
    return chunks


# ---------- step 3: keyword search (BM25) ----------
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
            break  # sorted descending: everything after this has no word match
        chunk = _corpus[idx]
        if source and chunk["source"] != source:
            continue
        hits.append({**chunk, "bm25": float(scores[idx])})
        if len(hits) == k:
            break
    return hits


# ---------- step 4: merge (reciprocal rank fusion) ----------
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


# ---------- step 5: rerank (CrossEncoder) ----------
_reranker = None


def _get_reranker() -> CrossEncoder:
    """Load lazily: the model is large, so only pay for it when first used."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, max_length=256)  # drop max_length if your version rejects it
    return _reranker


def rerank(query: str, chunks: list[dict], top_n: int = 4) -> list[dict]:
    """Score each (question, chunk) pair together; keep the best top_n."""
    if not chunks:
        return []
    scores = _get_reranker().predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [{**chunk, "rerank": float(score)} for chunk, score in ranked]


# ---------- steps 1-7 composed ----------
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


# ---------- what the model sees ----------
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


# ---------- sanity checks (run: python src/tools.py) ----------
if __name__ == "__main__":
    import time

    Q = "What are the core components of RAG?"        # a question whose answer you can find in the PDF
    RARE = "Core"       # an exact term or acronym that appears in the paper
    OFF = "recipe for pancakes"

    print("count:", col.count(), "| bm25 corpus:", len(_corpus))

    # 3c: metadata filter — every page printed must be 5
    print("3c pages:", [h["page"] for h in vector_search(Q, k=5, where={"page": 5})])

    # 3d: BM25 — top hit should literally contain the term
    kw = bm25_search(RARE, k=3)
    print("3d top hit contains term:", bool(kw) and RARE.lower() in kw[0]["text"].lower())

    # 3e: merge — ids unique, length <= 10
    merged = merge(vector_search(Q, k=10), bm25_search(Q, k=10), k=10)
    ids = [m["id"] for m in merged]
    print("3e unique ids:", len(ids) == len(set(ids)), "| length:", len(ids))

    # 3f: rerank — first run includes model load time; compare the SECOND run's time
    for label, q in (("relevant", Q), ("off-topic", OFF)):
        cands = merge(vector_search(q, k=10), bm25_search(q, k=10), k=10)
        t = time.time()
        top = rerank(q, cands, top_n=4)
        print(f"3f {label}: {time.time() - t:.1f}s |", [round(c["rerank"], 3) for c in top])

    # 3g: the tool — set MIN_RERANK_SCORE from 3f, then OFF should return the 'no passages' message
    print("3g tool:", retrieve.name)
    print(retrieve.invoke({"query": Q})[:600])
    print("3g off-topic:", retrieve.invoke({"query": OFF})[:200])