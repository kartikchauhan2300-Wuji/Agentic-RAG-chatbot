"""evals.py — evaluation harness for the research-paper agent.

Run from the project root:
    python src/evals.py retrieval          # no LLM; all questions
    python src/evals.py graph --limit 10   # full graph; slow
    python src/evals.py ragas --limit 5    # optional; local judge is noisy
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import tools

HERE = Path(__file__).resolve().parent
GOLDEN_PATH = HERE / "dataset.json"
RESULTS_DIR = HERE.parent / "eval_results"
PAGE_TOLERANCE = 0                  # 1 = a neighbouring page also counts as a hit
K_CANDIDATES, TOP_N = 10, 4
STAGES = ["vec", "bm25", "merged", "rerank", "final"]
REFUSAL_MARKERS = ("couldn't find", "couldn\u2019t find", "could not find", "needs at least two")


# ---------- helpers ----------
def load_golden():
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"Missing {GOLDEN_PATH}. Create it from EVALS.md section 1.")
    items = [i for i in json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
             if not i["question"].startswith("EDIT")]
    if not items:
        raise SystemExit("golden_set.json has no filled-in questions.")
    return items


def first_hit(pages, expected):
    """1-based rank of the first page that matches an expected page, else None."""
    for rank, page in enumerate(pages, 1):
        if page is not None and any(abs(page - e) <= PAGE_TOLERANCE for e in expected):
            return rank
    return None


def save_csv(rows, prefix):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", path)


# ---------- mode 1: retrieval, stage by stage (no LLM) ----------
def eval_retrieval(golden):
    thr = tools.MIN_RERANK_SCORE
    rows = []
    for i, item in enumerate(golden, 1):
        q, exp = item["question"], item.get("expected_pages", [])
        answerable = item.get("answerable", True)
        t = time.time()

        vec = tools.vector_search(q, k=K_CANDIDATES)
        kw = tools.bm25_search(q, k=K_CANDIDATES)
        merged = tools.merge(vec, kw, k=K_CANDIDATES)
        reranked = tools.rerank(q, merged, top_n=TOP_N)
        final = [c for c in reranked if thr is None or c["rerank"] >= thr]  # what the agent sees
        found = dict(zip(STAGES, [vec, kw, merged, reranked, final]))

        row = {"question": q, "answerable": answerable, "expected_pages": exp}
        for name, chunks in found.items():
            hit = first_hit([c["page"] for c in chunks], exp) is not None
            row[f"{name}_hit"] = hit if answerable else ""
        row.update(
            first_rank=first_hit([c["page"] for c in final], exp) if answerable else None,
            best_rerank_score=round(reranked[0]["rerank"], 3) if reranked else None,
            n_returned=len(final),
            refused=not final,
            seconds=round(time.time() - t, 1),
        )
        rows.append(row)
        print(f"[{i}/{len(golden)}] final_hit={row['final_hit']!s:5} "
              f"best={row['best_rerank_score']} refused={row['refused']} | {q[:55]}")
    return rows


def summarize_retrieval(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    print("\n===== RETRIEVAL SUMMARY =====")
    if ans:
        rate = {s: 100 * sum(r[f"{s}_hit"] is True for r in ans) / len(ans) for s in STAGES}
        mrr = sum(1 / r["first_rank"] for r in ans if r["first_rank"]) / len(ans)
        print(f"answerable questions: {len(ans)}")
        for s in STAGES:
            print(f"  {s:<7} hit rate: {rate[s]:.0f}%")
        print(f"  MRR (final list): {mrr:.2f}")
        if rate["rerank"] - rate["final"] >= 20:
            print("  -> hits lost AFTER reranking: the threshold is probably too high.")
        if rate["merged"] - rate["rerank"] >= 20:
            print("  -> hits lost AT reranking: the right page is pushed out of the top 4.")
        if rate["merged"] < 70:
            print("  -> the right page is often not a candidate: chunking or query wording.")
    if unans:
        refused = sum(r["refused"] for r in unans)
        print(f"unanswerable questions: {len(unans)} | correctly refused: {refused}")
        if refused < len(unans):
            print("  -> threshold too low: nonsense questions still return passages.")
    print(f"mean seconds per question: {sum(r['seconds'] for r in rows) / len(rows):.1f}")


# ---------- mode 2: the whole graph ----------
def eval_graph(golden, limit):
    from graph import ask   # imported here so 'retrieval' mode stays light

    rows = []
    for i, item in enumerate(golden[:limit], 1):
        q, exp = item["question"], item.get("expected_pages", [])
        answerable = item.get("answerable", True)
        want_route = item.get("expected_route", "")
        t = time.time()
        try:
            state = ask(q)
        except Exception as e:                       # record the failure and keep going
            state = {"answer": f"ERROR: {type(e).__name__}: {e}"}
        answer = state.get("answer", "")
        pages = [s.get("page") for s in state.get("sources", [])]
        refused = any(m in answer.lower() for m in REFUSAL_MARKERS)
        rows.append({
            "question": q,
            "expected_route": want_route,
            "route": state.get("route", ""),
            "route_ok": state.get("route") == want_route if want_route else "",
            "answerable": answerable,
            "refused": refused,
            "refusal_ok": refused == (not answerable),
            "source_pages": pages,
            "page_hit": first_hit(pages, exp) is not None if answerable else "",
            "seconds": round(time.time() - t, 1),
            "answer": answer.replace("\n", " ")[:600],
            "manual_grade": "",                      # fill in: correct / partial / wrong
            "notes": "",
        })
        print(f"[{i}/{min(limit, len(golden))}] route={rows[-1]['route']} "
              f"refused={refused} page_hit={rows[-1]['page_hit']} | {q[:55]}")
    return rows


def summarize_graph(rows):
    def pct(flags):
        return f"{100 * sum(flags) / len(flags):.0f}%" if flags else "n/a"

    print("\n===== GRAPH SUMMARY =====")
    print("route accuracy :", pct([r["route_ok"] for r in rows if r["route_ok"] != ""]))
    print("refusal correct:", pct([r["refusal_ok"] for r in rows]))
    print("source page hit:", pct([r["page_hit"] for r in rows if r["answerable"]]))
    print(f"mean seconds   : {sum(r['seconds'] for r in rows) / len(rows):.0f}")
    print("Now open the CSV and fill manual_grade for every answered question.")


# ---------- mode 3: RAGAS (optional, version-sensitive) ----------
def collect_ragas_samples(golden, limit):
    from graph import ask

    samples = []
    for item in golden:
        if len(samples) >= limit:
            break
        if not item.get("answerable", True) or not item.get("reference"):
            continue
        state = ask(item["question"])
        contexts = [c["text"] for c in state.get("chunks", [])]
        if contexts:
            samples.append({
                "user_input": item["question"],
                "retrieved_contexts": contexts,
                "response": state.get("answer", "").split("\n\nSources:")[0],  # drop sources block
                "reference": item["reference"],
            })
    return samples


def eval_ragas(golden, limit):
    try:                                             # check imports first: fail before the slow part
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                   LLMContextRecall, ResponseRelevancy)
        from ragas.run_config import RunConfig
        from graph import LLM_MODEL, OLLAMA_URL
    except ImportError as e:
        print("RAGAS import failed:", e)
        print("Run `pip install ragas`; if a name changed in your version, match the docs page you read.")
        return

    samples = collect_ragas_samples(golden, limit)
    if not samples:
        print("No usable samples (need answerable questions with a reference and retrieved chunks).")
        return

    judge = LangchainLLMWrapper(
        ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False))
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=tools.EMBED_MODEL, base_url=OLLAMA_URL))
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(),
                 LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=900, max_workers=1),    # one call at a time on CPU
    )
    df = result.to_pandas()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"ragas_{datetime.now():%Y%m%d_%H%M}.csv"
    df.to_csv(path, index=False)
    print(df.select_dtypes("number").mean())
    print("NaN = the judge failed on that item. Treat scores as rough; spot-check by hand.")
    print("saved:", path)


# ---------- entry point ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["retrieval", "graph", "ragas"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    golden = load_golden()

    if args.mode == "retrieval":
        rows = eval_retrieval(golden[: args.limit] if args.limit else golden)
        summarize_retrieval(rows)
        save_csv(rows, "retrieval")
    elif args.mode == "graph":
        rows = eval_graph(golden, args.limit or 10)
        summarize_graph(rows)
        save_csv(rows, "graph")
    else:
        eval_ragas(golden, args.limit or 5)


if __name__ == "__main__":
    main()