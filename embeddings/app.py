from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from time import time

from src.instructor import InstructorEmbedding

from settings import EMBEDDING_MODEL

app = FastAPI()

# init embedding model globally at startup so it can load before the first request
if EMBEDDING_MODEL.startswith("hkunlp/instructor"):
    embedding_model = InstructorEmbedding(EMBEDDING_MODEL)
else:
    raise ValueError(f"Unknown embedding model {EMBEDDING_MODEL}")

class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/embeddings")
def embed(
    data: EmbeddingRequest, 
):
    # needed for instruction embedding models
    instruction = "represent the document for retrieval"

    vectors = []
    embedding_start = time()
    # match openai spec (for langchain compatibility):
    # https://platform.openai.com/docs/api-reference/embeddings/object
    for idx, text in enumerate(data.input):
        vectors.append({
            "object": "embedding",
            "embedding": 
                embedding_model.embed(text, instruction),
            "index": idx,
        })
    embedding_end = time()
    logger.info(f"embedding took {embedding_end - embedding_start} seconds for {len(data.input)} documents.")

    # match openai spec (for langchain compatibility):
    # https://platform.openai.com/docs/api-reference/embeddings/create
    return {
        "object": "list",
        "data": vectors,
        "model": data.model,
    } 
