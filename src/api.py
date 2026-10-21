
import threading
import time
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import graph
import tools

# One question at a time: the CPU, the reranker and Ollama are shared resources.
_run_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    source: str
    page: int | None = None
    heading: str | None = None


class AskResponse(BaseModel):
    answer: str
    route: str | None
    sources: list[Source]
    seconds: float      # includes time spent waiting for the lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    tools._get_reranker()   # load the reranker once at startup so the first question isn't slow
    yield


app = FastAPI(title="Research Paper Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": tools.col.count(), "papers": graph.list_sources()}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    t = time.time()
    with _run_lock:
        try:
            state = graph.ask(req.question)
        except httpx.ConnectError:
            raise HTTPException(503, f"Ollama is not reachable at {graph.Ollama_url}. Start it and retry.")
        except GraphRecursionError:
            raise HTTPException(500, "The agent hit its step limit without finishing.")
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")
    return AskResponse(
        answer=state.get("answer", ""),
        route=state.get("route"),
        sources=[Source(**s) for s in state.get("sources", [])],
        seconds=round(time.time() - t, 1),

  )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)