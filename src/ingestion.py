from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from tqdm import tqdm
import ollama
import numpy as np
import chromadb
source = "https://arxiv.org/pdf/2501.09136"
converter = DocumentConverter()
doc = converter.convert(source).document


tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5"),
    max_tokens=512
)
chunker = HybridChunker(tokenizer=tokenizer)
chunks = list(chunker.chunk(doc))
def embed_chunk(text: str) -> list[float]:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

vectors = [embed_chunk(chunk.text) for chunk in tqdm(chunks)]
print(np.linalg.norm(vectors[0]))

client = chromadb.PersistentClient(path = "./chromadb")
col = client.get_or_create_collection(
    name = "rag_docs",
    metadata = {"hnsw:space": "cosine"}
)
ids = [f"chunk_{i}" for i in range(len(chunks))]
documents = [chunk.text for chunk in chunks]

metadatas = []
for chunk in chunks:
    item = chunk.meta.doc_items[0] if chunk.meta.doc_items else None
    page_no = item.prov[0].page_no if item and item.prov else None
    heading = chunk.meta.headings[0] if chunk.meta.headings else "unknown"

    metadatas.append({
        "page": page_no,
        "heading": heading,
        "source": chunk.meta.origin.filename if chunk.meta.origin else "unknown",
    })

col.add(
    documents=documents,
    embeddings=vectors,
    ids=ids,
    metadatas=metadatas,
)
print(metadatas(0))
print(metadatas(50))
