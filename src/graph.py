import operator
from typing import Annotated, TypedDict, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
import tools
from pathlib import Path
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from schemas import(
    RouteDecision,
    GradeDocuments,
    PaperSummary,
    Citation,
    PaperComparison,
    ComparisonEntry,
)
LLM_MODEL =   "qwen3:8b"
Ollama_url = "http://127.0.0.1:11434"
max_retries = 2
use_llm_grader = True
grader_chars = 600
CACHE_PATH = Path(__file__).resolve().parent/ "cache" / "llm_cache.db"
def enable_cache(on: bool = True) -> None:
    """Turn the SQLite cache on (survives restarts) or off (always call the model)."""
    if on:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(CACHE_PATH)))
    else:
        set_llm_cache(None)


def clear_cache() -> None:
    """Delete every stored response, whether or not the cache is currently on."""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    SQLiteCache(database_path=str(CACHE_PATH)).clear()


enable_cache(True)
llm = ChatOllama(model = LLM_MODEL, base_url = Ollama_url, reasoning = False, temperature= 0)
router_llm = llm.with_structured_output(RouteDecision)
grader_llm = llm.with_structured_output(GradeDocuments)
extract_llm = llm.with_structured_output(PaperSummary)

class GraphState(TypedDict, total = False):
    question:str
    query:str
    route:str
    chunks:list[dict]
    retries:int
    relevant:bool
    answer:str
    sources: list[dict]
    summary: dict | None
    comparison: dict | None
    comparison_entries: Annotated[list[dict], operator.add]

class WorkerState(TypedDict):
    question:str
    source:str

def _safe(structured_llm, prompt):
    """Structured calls can fail on a small local model; return None instead of crashing."""
    try:
        return structured_llm.invoke(prompt)
    except Exception as e:
        print(f"[structured call failed] {type(e).__name__}: {e}")
        return None


def _citations(chunks: list[dict]) -> list[dict]:
    """Citations come from chunk metadata (code), never from the model."""
    seen, out = set(), []
    for c in chunks:
        key = (c["source"], c["page"], c["heading"])
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(
            source=c["source"] or "unknown",
            page=c["page"],
            heading=c["heading"] or "unknown",
        ).model_dump())
    return out


def _sources_block(sources: list[dict]) -> str:
    lines = [f"- {s['source']}, p.{s['page']}, {s['heading']}" for s in sources]
    return "Sources:\n" + "\n".join(lines)


def list_sources() -> list[str]:
    """Distinct papers currently indexed (read at call time, so re-ingestion is picked up)."""
    return sorted({c["source"] for c in tools._corpus if c["source"]})


def retrieve_node(state: GraphState):
    return {"chunks": tools.retrieve_chunks(state["query"])}


def grade_node(state: GraphState):
    chunks = state.get("chunks", [])
    if not chunks:  # nothing cleared the reranker threshold
        return {"relevant": False}
    if not use_llm_grader:
        return {"relevant": True}
    passages = "\n\n".join(c["text"][:grader_chars] for c in chunks[:3])
    prompt = f"""Decide whether the passages contain information that helps answer the question.
Answer 'yes' if they do, otherwise 'no'.

Question: {state['question']}

Passages:
{passages}"""
    grade = _safe(grader_llm, prompt)
    # if the grader itself fails, trust the reranker (chunks already passed its threshold)
    return {"relevant": (grade.binary_score == "yes") if grade else True}


def decide_after_grade(state: GraphState) -> Literal["rewrite", "give_up", "answer", "extract"]:
    if state.get("relevant"):
        return "extract" if state["route"] == "extract" else "answer"
    if state.get("retries", 0) >= max_retries:
        return "give_up"
    return "rewrite"


def rewrite_node(state: GraphState):
    prompt = f"""This search query did not retrieve good passages from research papers.
Rewrite it as a more specific search query using the technical terms a paper would use.
Return only the rewritten query, nothing else.

Original question: {state['question']}
Last query: {state['query']}"""
    text = llm.invoke(prompt).content.strip().strip('"')
    new_query = text.splitlines()[0] if text else state["question"]
    return {"query": new_query or state["question"], "retries": state.get("retries", 0) + 1}


# ---------- nodes: final answers ----------
def answer_node(state: GraphState):
    chunks = state["chunks"]
    prompt = f"""Answer the question using ONLY the passages below.
If they do not contain the answer, say you could not find it. Be concise. Do not state page numbers; Sources are added seperately.

Passages:
{tools.format_chunks(chunks)}

Question: {state['question']}"""
    text = llm.invoke(prompt).content.strip()
    sources = _citations(chunks)
    return {"answer": f"{text}\n\n{_sources_block(sources)}", "sources": sources}


def extract_node(state: GraphState):
    # NOTE: one chunk, as in the validated Stage 2 test. Facts spread across chunks will be missed.
    chunk = state["chunks"][0]
    prompt = f"""Extract the following specific facts from this text. Pay particular
attention to naming the dataset explicitly if one is mentioned by name
(e.g. SQuAD, GLUE, ImageNet) — do not fold it into another field.

Section heading: {chunk['heading']}

Text:
{chunk['text']}"""
    summary = _safe(extract_llm, prompt)
    sources = _citations([chunk])
    if summary is None:
        return {
            "answer": "I found relevant text but could not extract structured facts from it.",
            "sources": sources,
            "summary": None,
        }
    text = (
        f"Dataset: {summary.dataset_used or 'not stated'}\n"
        f"Method: {summary.methodology}\n"
        f"Key result: {summary.key_result}"
    )
    return {
        "answer": f"{text}\n\n{_sources_block(sources)}",
        "sources": sources,
        "summary": summary.model_dump(),
    }


def give_up_node(state: GraphState):
    return {"answer": "I couldn't find this in the indexed papers.", "sources": []}


def not_enough_papers_node(state: GraphState):
    n = len(list_sources())
    return {
        "answer": f"Comparison needs at least two indexed papers; only {n} found.",
        "sources": [],
    }


# ---------- nodes: comparison (orchestrator-worker) ----------
def compare_worker(state: WorkerState):
    chunks = tools.retrieve_chunks(state["question"], source=state["source"], top_n=2)
    if not chunks:
        finding, page = "No relevant passage found.", None
    else:
        prompt = f"""In one or two sentences, state what this paper says about the question.
Use only the passages. If they do not cover it, reply exactly: Not covered.

Passages:
{tools.format_chunks(chunks)}

Question: {state['question']}"""
        finding = llm.invoke(prompt).content.strip()
        page = chunks[0]["page"]
    # source and page are filled by code, not by the model
    entry = ComparisonEntry(source=state["source"], finding=finding, page=page)
    return {"comparison_entries": [entry.model_dump()]}

def route_question(state: GraphState):
    prompt = f"""Classify this question about research papers into exactly one route.
- qa: asks what a paper says (concepts, explanations, how something works) where something is discussed, difference between ideas within a paper.
- extract: ask for paper's dataset, method and main result as structured facts
- compare: asks to compare What different papers say (needs more than one paper)

Question: {state['question']}"""
    decision = _safe(router_llm, prompt)
    route = decision.route if decision else "qa"   # safe default if the call fails
    return {"route": route, "query": state["question"], "retries": 0}

def dispatch_route(state: GraphState):
    if state["route"] == "compare" and len(list_sources()) >= 2:
        return [
            Send("compare_worker", {"question": state["question"], "source": s})
            for s in list_sources()
        ]
    return "retrieve"   # qa, extract, and compare-with-one-paper

def synthesize_node(state: GraphState):
    entries = state["comparison_entries"]
    listing = "\n".join(f"- {e['source']}: {e['finding']}" for e in entries)
    prompt = f"""Compare how these papers address the question. Use only the findings given.
Write two or three sentences on similarities and differences.

Question: {state['question']}

Findings:
{listing}"""
    summary = llm.invoke(prompt).content.strip()
    comparison = PaperComparison(
        aspect=state["question"],
        entries=[ComparisonEntry(**e) for e in entries],
        summary=summary,
    )
    lines = "\n".join(f"- {e['source']} (p.{e['page']}): {e['finding']}" for e in entries)
    return {
        "comparison": comparison.model_dump(),
        "answer": f"{summary}\n\n{lines}",
        "sources": [{"source": e["source"], "page": e["page"]} for e in entries],
    }


# ---------- build the graph ----------
builder = StateGraph(GraphState)

builder.add_node("route_question", route_question)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("answer", answer_node)
builder.add_node("extract", extract_node)
builder.add_node("give_up", give_up_node)
builder.add_node("not_enough_papers", not_enough_papers_node)
builder.add_node("compare_worker", compare_worker)
builder.add_node("synthesize", synthesize_node)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question", dispatch_route, ["retrieve", "not_enough_papers", "compare_worker"]
)
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade", decide_after_grade, ["rewrite", "give_up", "answer", "extract"]
)
builder.add_edge("rewrite", "retrieve")  # the capped retry loop
builder.add_edge("answer", END)
builder.add_edge("extract", END)
builder.add_edge("give_up", END)
builder.add_edge("not_enough_papers", END)
builder.add_edge("compare_worker", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()


def ask(question: str, limit: int = 25) -> dict:
    """Run the whole graph and return the final state."""
    return graph.invoke({"question": question}, {"recursion_limit": limit})
def trace(question: str, limit: int = 25) -> None:
    """Print which nodes ran, in order."""
    from langgraph.errors import GraphRecursionError

    try:
        for step in graph.stream(
            {"question": question},
            {"recursion_limit": limit},
            stream_mode="updates",
        ):
            print("NODE:", list(step.keys())[0])
    except GraphRecursionError:
        print("HIT RECURSION LIMIT")


def route_question(state: GraphState):
    prompt = f"""Classify this question about research papers into exactly one route.
    - qa: asks what a paper says (concepts, explanations, how something works)
    - extract: asks for specific facts such as the dataset used, the method, or the main result
    - compare: asks to compare what DIFFERENT PAPERS say (only when several papers are being compared)

    Question: {state['question']}"""
    decision = _safe(router_llm, prompt)
    route = decision.route if decision else "qa"  # safe default if the call fails
    return {"route": route, "query": state["question"], "retries": 0}

if __name__ == "__main__":
    # Each check can take minutes on CPU. Comment out the ones you are not running.
    Q = "What are the core components of RAG?"
    OFF = "What is the capital of France?"

    # 4a: router — edit the expected labels; run twice; aim for 5/6 and the same label both times
    router_tests = [
        ("What are vector databases?", "qa"), ("What is Agentic RAG?", "qa"),
        ("What is the method required in an Agentic RAG pipeline?", "extract"), ("On what pages are use cases about RAG given?", "extract"),
        ("Compare the methods given in this paper with other sources", "compare"), ("Compare the difference between info about Naive RAG from this data and other sources", "compare"),
    ]
    for q, expected in router_tests:
        got = [route_question({"question": q})["route"] for _ in range(2)]
        print("4a", "OK " if got == [expected, expected] else "BAD", expected, got, "|", q[:50])

    # 4b: grader — relevant text should be yes, junk should be no
    print("4b relevant:", grade_node({"question": Q, "chunks": [{"text": "Agentic RAG employs iterative refinement and adaptive retrieval strategies to address complex, real-time and multi-domain queries."}]}))
    print("4b junk    :", grade_node({"question": Q, "chunks": [{"text": "meow"}]}))

    # 4c: rewrite — non-empty, different from the original, same meaning
    print("4c:", rewrite_node({"question": Q, "query": Q, "retries": 0}))

    # 4d: retry cap — must END with the 'couldn't find' answer, never HIT RECURSION LIMIT
    trace(OFF)
    print("4d answer:", ask(OFF)["answer"])

    # 4e: full QA path — fact present, citation page/heading match the PDF
    trace(Q)
    print("4e answer:", ask(Q)["answer"])

    # 4g: compare with one paper indexed — should say it needs two, not crash
    print("4g:", ask("Compare the methods of the indexed papers")["answer"])

