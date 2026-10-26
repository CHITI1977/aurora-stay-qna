
# Import necessary libraries
import logging
import time

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import RAG utility functions
from rag_utils import retrieve, rag, rag_with_sources


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"
MAX_QUERY_CHARS = 1000      # Reject unreasonably long inputs
MAX_K = 10                  # Upper bound on chunks per request
MAX_TOKENS_LIMIT = 2048     # Upper bound on answer length


# ---------------------------------------------------------
# Logging (written to the container log, visible via `docker logs backend`)
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("aurorastay-api")


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model and the saved FAISS vector store once at startup
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")
logger.info("Loaded FAISS index with %s vectors", vectorstore.index.ntotal)


# ---------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------

class ValidationError(Exception):
    pass


def parse_request(require_generation_params: bool):
    """Validate the JSON body and return a dict of clean parameters."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object.")

    # Accept both "query" (template contract) and "question" (/query contract)
    query = data.get("query", data.get("question"))
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("'query' must be a non-empty string.")
    if len(query) > MAX_QUERY_CHARS:
        raise ValidationError(f"'query' must be at most {MAX_QUERY_CHARS} characters.")

    try:
        k = int(data.get("k", 3))
    except (TypeError, ValueError):
        raise ValidationError("'k' must be an integer.")
    if not 1 <= k <= MAX_K:
        raise ValidationError(f"'k' must be between 1 and {MAX_K}.")

    params = {"query": query.strip(), "k": k}

    if require_generation_params:
        try:
            params["model_name"] = str(data.get("model_name", "gpt-4o-mini"))
            params["temperature"] = float(data.get("temperature", 0.0))
            params["top_p"] = float(data.get("top_p", 1.0))
            params["max_tokens"] = int(data.get("max_tokens", 512))
        except (TypeError, ValueError):
            raise ValidationError("temperature/top_p must be numbers and max_tokens an integer.")
        if not 0.0 <= params["temperature"] <= 2.0:
            raise ValidationError("'temperature' must be between 0 and 2.")
        if not 0.0 < params["top_p"] <= 1.0:
            raise ValidationError("'top_p' must be in (0, 1].")
        if not 16 <= params["max_tokens"] <= MAX_TOKENS_LIMIT:
            raise ValidationError(f"'max_tokens' must be between 16 and {MAX_TOKENS_LIMIT}.")

    return params


@rag_api.errorhandler(ValidationError)
def handle_validation_error(err):
    logger.warning("Rejected request: %s", err)
    return jsonify({"error": str(err)}), 400


@rag_api.errorhandler(Exception)
def handle_unexpected_error(err):
    # Let normal HTTP errors (404, 405, ...) pass through with their own status code
    if isinstance(err, HTTPException):
        return jsonify({"error": err.description}), err.code
    logger.exception("Unhandled error")
    return jsonify({"error": "Internal server error. Check the backend logs."}), 500


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Health endpoint (used by the frontend status indicator and for monitoring)
# ---------------------------------------------------------

@rag_api.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "embedding_model": EMBED_MODEL,
        "vectors": int(vectorstore.index.ntotal),
    })


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Example request: {"query": "What is the cancellation policy?", "k": 3}
    Returns the top-k relevant chunks retrieved from the vector store.
    """
    p = parse_request(require_generation_params=False)
    logger.info("relevant_chunks | k=%s | query=%r", p["k"], p["query"])

    chunks = retrieve(query=p["query"], k=p["k"], vectorstore=vectorstore)

    return jsonify({
        "query": p["query"],
        "k": p["k"],
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Example request:
    {"query": "...", "k": 3, "model_name": "gpt-4o-mini",
     "temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
    Returns both the generated answer and the relevant chunks used to generate it.
    """
    p = parse_request(require_generation_params=True)
    start = time.time()

    answer, chunks = rag(
        query=p["query"],
        k=p["k"],
        model_name=p["model_name"],
        temperature=p["temperature"],
        top_p=p["top_p"],
        max_tokens=p["max_tokens"],
        vectorstore=vectorstore
    )
    logger.info("answer_with_relevant_chunks | %.1fs | query=%r", time.time() - start, p["query"])

    return jsonify({
        "query": p["query"],
        "k": p["k"],
        "model_name": p["model_name"],
        "temperature": p["temperature"],
        "top_p": p["top_p"],
        "max_tokens": p["max_tokens"],
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Same request body as /v1/answer_with_relevant_chunks. Returns only the generated answer.
    """
    p = parse_request(require_generation_params=True)
    start = time.time()

    generated_answer, _ = rag(
        query=p["query"],
        k=p["k"],
        model_name=p["model_name"],
        temperature=p["temperature"],
        top_p=p["top_p"],
        max_tokens=p["max_tokens"],
        vectorstore=vectorstore
    )
    logger.info("answer | %.1fs | query=%r", time.time() - start, p["query"])

    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Structured query endpoint: answer + source documents + scored chunks + confidence
# ---------------------------------------------------------

@rag_api.post("/query")
def query():
    """
    Example request: {"question": "Is breakfast free at Miami Brickell?", "k": 3}
    Response: {"question", "answer", "sources", "retrieved_chunks", "confidence", "refused"}
    """
    p = parse_request(require_generation_params=True)
    start = time.time()

    result = rag_with_sources(
        query=p["query"],
        k=p["k"],
        model_name=p["model_name"],
        temperature=p["temperature"],
        top_p=p["top_p"],
        max_tokens=p["max_tokens"],
        vectorstore=vectorstore
    )
    latency = round(time.time() - start, 2)
    logger.info("query | %.1fs | confidence=%s | sources=%s | question=%r",
                latency, result["confidence"], result["sources"], p["query"])

    return jsonify({
        "question": p["query"],
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieved_chunks": result["retrieved_chunks"],
        "confidence": result["confidence"],
        "refused": result["refused"],
        "latency_seconds": latency,
    })


# ---------------------------------------------------------
# Run the Flask application (local debugging only; the container uses gunicorn)
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=True)
