"""Azure Functions v2 entrypoint – replaces the old FastAPI main.py.

Three HTTP triggers that mirror the original REST API:
  GET  /api/health  → health check
  POST /api/chat    → RAG pipeline + LLM answer
  POST /api/ingest  → refresh URL context
"""

import json
import logging
import os

import azure.functions as func
from dotenv import load_dotenv

# Load .env for local development (Azure provides env vars in production)
load_dotenv()

from rag.llm import generate_answer
from rag.prompt_builder import build_prompt
from rag.web_context import get_url_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------
@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps({"status": "ok"}),
        status_code=200,
        mimetype="application/json",
    )


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------
@app.route(route="chat", methods=["POST"])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    """Chat using live URL context instead of PDF embeddings."""
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON body"}),
            status_code=400,
            mimetype="application/json",
        )

    question = body.get("question")
    if not question:
        return func.HttpResponse(
            json.dumps({"error": "Missing 'question' field"}),
            status_code=400,
            mimetype="application/json",
        )

    history = body.get("history", [])

    try:
        source_chunks = get_url_context()
        prompt = build_prompt(question, source_chunks, history)
        answer = generate_answer(prompt)
        sources = [chunk["source"] for chunk in source_chunks]

        return func.HttpResponse(
            json.dumps({"answer": answer, "sources": sources}),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as exc:
        logger.error("Error in chat function: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": f"Error processing query: {str(exc)}"}),
            status_code=500,
            mimetype="application/json",
        )


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------
@app.route(route="ingest", methods=["POST"])
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    """Refresh configured URL context (kept for compatibility)."""
    try:
        source_chunks = get_url_context(force_refresh=True)
        sources = [chunk["source"] for chunk in source_chunks]

        return func.HttpResponse(
            json.dumps({"sources_loaded": len(source_chunks), "sources": sources}),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as exc:
        logger.error("Error refreshing URL context: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": f"URL refresh failed: {str(exc)}"}),
            status_code=500,
            mimetype="application/json",
        )
