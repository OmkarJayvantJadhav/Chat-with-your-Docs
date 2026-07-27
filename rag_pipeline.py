"""
rag_pipeline.py — Core RAG logic for "Chat With Your Docs"

Handles PDF text extraction, chunking, embedding, vector storage/retrieval,
prompt assembly, and LLM generation via Groq cloud API.
"""

from __future__ import annotations

# Prevent transformers from importing TensorFlow/Keras (we only need PyTorch)
# Must be set before any transformers/sentence_transformers imports
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import hashlib
import io
import json
import re
from typing import Any, Optional, Generator

import time
import chromadb
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configurable constants — tweak these, don't hardcode values inline
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 1000
"""Target size (in characters) for each text chunk."""

CHUNK_OVERLAP: int = 200
"""Number of overlapping characters between consecutive chunks."""

TOP_K: int = 10
"""Number of top chunks to retrieve for each query."""

DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"
"""Default model for Groq cloud inference."""

GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
"""Groq REST API endpoint (OpenAI-compatible)."""

CHROMA_DB_PATH: str = "./chroma_db"
"""Path for persistent ChromaDB storage."""

COLLECTION_NAME: str = "documents"
"""Name of the ChromaDB collection."""

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
"""Sentence-transformers model used for embedding."""

SYSTEM_PROMPT: str = (
    "You are an expert, highly intelligent AI research assistant designed to analyze and summarize uploaded documents. "
    "Answer the user's question comprehensively and accurately using ONLY the context provided below. "
    "Structure your response clearly using markdown, bullet points, and code blocks where appropriate to make it as helpful and readable as possible. "
    "When asked about sequential items (e.g. 'first code', 'first problem', 'beginning', 'start'), carefully examine the earliest document sections and chunk order provided in the context to identify the chronological start. "
    "If the user asks a broad question like 'what is this document about?', 'summarize this PDF', or asks for an overview, synthesize a thorough, professional overview of the document's core concepts, algorithms, and themes. "
    "If the provided context lacks the specific facts needed to answer a factual query, explain clearly what related information is present in the context without guessing. "
    "Do not make up information or hallucinate external facts."
)
"""Anti-hallucination system prompt — the core grounding mechanism."""

# ---------------------------------------------------------------------------
# Module-level cache for the embedding model
# ---------------------------------------------------------------------------

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection: Optional[chromadb.Collection] = None


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file: io.BytesIO | Any) -> str:
    """Extract all text content from a PDF file.

    Args:
        file: A file-like object (BytesIO or UploadedFile) containing PDF data.

    Returns:
        Concatenated text from all pages. Returns an empty string if the PDF
        is scanned/image-only or contains no extractable text.
    """
    try:
        reader = PdfReader(file)
        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries.

    Splits on paragraph breaks (\\n\\n), then line breaks (\\n), then
    sentence endings (. ), then word boundaries — in that priority order.
    Never splits mid-word when avoidable.

    Args:
        text: The full text to split.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks. Returns an empty list for empty/whitespace input.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # If the entire text fits in one chunk, return it directly
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we've reached or passed the end of text, take the rest
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find the best split point within the chunk window
        segment = text[start:end]
        split_pos = _find_best_split(segment)

        chunk = text[start : start + split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        next_start = start + split_pos - overlap
        if next_start > start and next_start < len(text):
            # Ensure next_start is at a word boundary (character after whitespace)
            adj_forward = next_start
            while adj_forward < start + split_pos and text[adj_forward - 1] not in " \n\t\r":
                adj_forward += 1

            adj_backward = next_start
            while adj_backward > start and text[adj_backward - 1] not in " \n\t\r":
                adj_backward -= 1

            if adj_forward < start + split_pos and (adj_forward - next_start) <= (next_start - adj_backward):
                next_start = adj_forward
            elif adj_backward > start:
                next_start = adj_backward
            else:
                next_start = adj_forward

        # Ensure we always make forward progress
        if next_start <= start:
            next_start = start + split_pos
        start = next_start

    return chunks


def _find_best_split(segment: str) -> int:
    """Find the best position to split a text segment, preferring natural boundaries.

    Tries paragraph breaks, line breaks, sentence endings, then word boundaries
    in priority order. Searches the last 30% of the segment to keep chunks
    reasonably sized.

    Args:
        segment: The text segment to find a split point in.

    Returns:
        The character index at which to split (relative to segment start).
    """
    search_start = int(len(segment) * 0.7)
    search_zone = segment[search_start:]

    # Priority 1: Paragraph break
    pos = search_zone.rfind("\n\n")
    if pos != -1:
        return search_start + pos + 2  # Split after the double newline

    # Priority 2: Line break
    pos = search_zone.rfind("\n")
    if pos != -1:
        return search_start + pos + 1

    # Priority 3: Sentence ending (. followed by space or end)
    for pattern in [". ", "! ", "? "]:
        pos = search_zone.rfind(pattern)
        if pos != -1:
            return search_start + pos + len(pattern)

    # Priority 4: Word boundary in search zone
    pos = search_zone.rfind(" ")
    if pos != -1:
        return search_start + pos + 1

    # Priority 5: Word boundary anywhere in segment before falling back
    pos = segment.rfind(" ")
    if pos != -1 and pos > 0:
        return pos + 1

    # Fallback: hard split at chunk_size
    return len(segment)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def get_embedding_function() -> SentenceTransformer:
    """Load and cache the sentence-transformer embedding model.

    Returns:
        A SentenceTransformer instance for generating embeddings.
        The model is loaded once and reused on subsequent calls.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str], model: SentenceTransformer | None = None) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        model: Optional pre-loaded SentenceTransformer. If None, loads the default.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if model is None:
        model = get_embedding_function()
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------


def get_chroma_collection(
    path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create a persistent ChromaDB collection.

    Args:
        path: Filesystem path for the persistent database.
        collection_name: Name of the collection to use.

    Returns:
        A ChromaDB Collection object backed by persistent storage.
    """
    global _chroma_client, _chroma_collection
    if _chroma_collection is None or _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=path)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def add_documents(
    chunks: list[str],
    filename: str,
    collection: chromadb.Collection,
    embed_fn: SentenceTransformer | None = None,
) -> int:
    """Embed text chunks and add them to the ChromaDB collection.

    Each chunk is stored with metadata including the source filename and
    chunk index. Document IDs are deterministic hashes to avoid duplicates.

    Args:
        chunks: List of text chunks to embed and store.
        filename: Source filename for metadata tagging.
        collection: The ChromaDB collection to add documents to.
        embed_fn: Optional pre-loaded SentenceTransformer model.

    Returns:
        The number of chunks added.
    """
    if not chunks:
        return 0

    embeddings = embed_texts(chunks, embed_fn)

    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        # Deterministic ID from filename + chunk index to avoid duplicates
        doc_id = hashlib.md5(f"{filename}_{i}_{chunk[:50]}".encode()).hexdigest()
        ids.append(doc_id)
        metadatas.append({
            "source": filename,
            "chunk_index": i,
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def query_documents(
    question: str,
    collection: chromadb.Collection,
    embed_fn: SentenceTransformer | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Query the vector store for the most relevant chunks.

    Args:
        question: The user's question to find relevant context for.
        collection: The ChromaDB collection to search.
        embed_fn: Optional pre-loaded SentenceTransformer model.
        top_k: Number of top results to return.

    Returns:
        A dict with keys 'documents', 'metadatas', and 'distances',
        each containing a list (of lists) of results. Returns empty
        structure if collection is empty.
    """
    count = collection.count()
    if count == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Don't request more results than exist
    effective_k = min(top_k, count)

    query_embedding = embed_texts([question], embed_fn)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    # For beginning/first/summary queries, ensure the first several chunks (0 to 5) are included
    if any(w in question.lower() for w in ["first", "start", "begin", "1st", "initial", "one", "about", "summary", "summarize", "overview", "what is", "what are", "who", "explain", "describe", "title", "intro", "problem 1", "code 1"]):
        try:
            intro_chunks = collection.get(where={"chunk_index": {"$in": [0, 1, 2, 3, 4, 5]}}, include=["documents", "metadatas"])
            if intro_chunks and intro_chunks.get("documents"):
                existing_docs = set(results["documents"][0]) if results.get("documents") and results["documents"][0] else set()
                for doc, meta in zip(intro_chunks["documents"], intro_chunks["metadatas"]):
                    if doc and doc not in existing_docs and results.get("documents") and len(results["documents"]) > 0:
                        results["documents"][0].append(doc)
                        results["metadatas"][0].append(meta)
                        results["distances"][0].append(0.0)
        except Exception:
            pass

    # Sort retrieved chunks chronologically by document order so the LLM reads a coherent, unfragmented narrative!
    if results.get("documents") and results["documents"] and results["documents"][0]:
        try:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            combined = list(zip(docs, metas, dists))
            combined.sort(key=lambda x: (x[1].get("source", "") if x[1] else "", x[1].get("chunk_index", 0) if x[1] else 0))
            results["documents"][0] = [item[0] for item in combined][:effective_k]
            results["metadatas"][0] = [item[1] for item in combined][:effective_k]
            results["distances"][0] = [item[2] for item in combined][:effective_k]
        except Exception:
            pass

    return results


def clear_collection(
    path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Delete and recreate the ChromaDB collection, effectively clearing all documents.

    Args:
        path: Filesystem path for the persistent database.
        collection_name: Name of the collection to clear.
    """
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=path)
    try:
        _chroma_client.delete_collection(name=collection_name)
    except ValueError:
        pass  # Collection doesn't exist, nothing to delete
    # Recreate empty collection
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection_stats(
    path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    """Get statistics about the current document collection.

    Args:
        path: Filesystem path for the persistent database.
        collection_name: Name of the collection.

    Returns:
        A dict with 'total_chunks' and 'sources' (unique filenames).
    """
    try:
        collection = get_chroma_collection(path, collection_name)
        count = collection.count()

        sources: set[str] = set()
        if count > 0:
            # Fetch all metadata to get unique sources
            all_data = collection.get(include=["metadatas"])
            if all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])

        return {
            "total_chunks": count,
            "sources": sorted(sources),
        }
    except Exception:
        return {"total_chunks": 0, "sources": []}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_prompt(question: str, retrieved_chunks: list[str]) -> str:
    """Assemble a grounded prompt from retrieved context and the user's question.

    Args:
        question: The user's question.
        retrieved_chunks: List of relevant text chunks from the vector store.

    Returns:
        A formatted prompt string including the system instruction, context
        chunks, and the user's question.
    """
    if not retrieved_chunks:
        context = "No relevant context was found in the uploaded documents."
    else:
        context_parts: list[str] = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}]\n{chunk}")
        context = "\n\n".join(context_parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM generation — Groq (cloud) and Ollama (local)
# ---------------------------------------------------------------------------


def rewrite_query_with_ai(
    question: str,
    chat_history: list[dict] | None = None,
    api_key: str = "",
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    """Rewrite a user's natural language question into rich technical search keywords
    using Groq API, resolving any pronouns or references from chat history.
    """
    if not api_key:
        return question

    history_text = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-4:]  # last 2 turns
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content and not content.startswith("⚠️"):
                history_text += f"{role.upper()}: {content[:300]}\n"

    system_prompt = (
        "You are an AI search query optimizer for a document retrieval database. "
        "Your job is to take the user's latest question (and recent conversation history if provided) "
        "and rewrite it into 5-10 specific, professional, technical keywords and concepts likely to appear "
        "in the target document. Resolve any pronouns like 'it', 'that', 'the first one'. "
        "Output ONLY the raw search keywords separated by spaces. Do not output markdown, quotes, or explanations."
    )
    user_prompt = f"Recent History:\n{history_text}\nLatest Question: {question}" if history_text else f"Question: {question}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 60,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        res = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            rewritten = res.json()["choices"][0]["message"]["content"].strip()
            rewritten = re.sub(r'^["\']|["\']$', '', rewritten)
            if len(rewritten) > 3 and "here are" not in rewritten.lower():
                return f"{question} {rewritten}"
    except Exception:
        pass
    return question


def generate_document_questions(
    context_chunks: list[str],
    api_key: str = "",
    model: str = DEFAULT_GROQ_MODEL,
) -> list[str]:
    """Analyze document chunks using Groq API to generate 3 custom, highly relevant
    suggested questions tailored specifically to the document's content.
    """
    default_questions = [
        "💡 What is the main topic and executive summary of this document?",
        "🔍 What are the key problems, algorithms, or concepts discussed?",
        "📝 Summarize the most important findings and conclusions",
    ]
    if not api_key or not context_chunks:
        return default_questions

    sample_text = "\n\n".join(context_chunks[:5])[:3000]

    system_prompt = (
        "You are an AI document analyst. Read the provided document excerpt and generate exactly 3 "
        "specific, fascinating, highly relevant questions that a user would want to ask about this specific document. "
        "Each question should start with an appropriate emoji (like 💡, 🔍, 📝, ⚙️, 🚀, ⚖️, 📊). "
        "Format your output as exactly 3 lines, one question per line. Do not number them or add extra text."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document Excerpt:\n{sample_text}"},
        ],
        "temperature": 0.4,
        "max_tokens": 150,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        res = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            lines = [line.strip() for line in res.json()["choices"][0]["message"]["content"].split("\n") if line.strip()]
            questions = [l for l in lines if len(l) > 10 and not l.lower().startswith("here")]
            if len(questions) >= 3:
                return questions[:3]
    except Exception:
        pass
    return default_questions


def generate_answer(
    question: str,
    context_chunks: list[str],
    model: str = "",
    api_key: str = "",
) -> str:
    """Generate a complete (non-streaming) answer using Groq.

    Args:
        question: The user's question.
        context_chunks: Retrieved context chunks to ground the answer.
        model: Groq model name (uses default if empty).
        api_key: Groq API key.

    Returns:
        The generated answer text.
    """
    effective_model = model or DEFAULT_GROQ_MODEL
    if not api_key:
        raise ValueError("Groq API key is required. Get a free key at https://console.groq.com")

    prompt_content = build_prompt(question, context_chunks)
    payload = {
        "model": effective_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ],
        "temperature": 0.2,
        "max_tokens": 1536,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(2):
        try:
            response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
        except requests.ConnectionError:
            raise ConnectionError("Could not connect to Groq API. Check your internet connection.")
        except requests.Timeout:
            raise ConnectionError("Groq API request timed out. Please try again.")

        if response.status_code == 429:
            if attempt == 0:
                wait_time = 10.0
                try:
                    err_msg = response.json().get("error", {}).get("message", "")
                    if "try again in" in err_msg:
                        parts = err_msg.split("try again in ")[1].split("s")[0]
                        wait_time = min(float(parts) + 1.0, 15.0)
                except Exception:
                    pass
                time.sleep(wait_time)
                continue
            raise ConnectionError(
                "Groq rate limit reached. Please wait ~10 seconds and try again."
            )
        if response.status_code == 401:
            raise ValueError("Invalid Groq API key. Check your key at https://console.groq.com")
        if response.status_code != 200:
            raise ConnectionError(f"Groq API error (HTTP {response.status_code}): {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    return ""


def stream_answer_groq(
    question: str,
    context_chunks: list[str],
    model: str = DEFAULT_GROQ_MODEL,
    api_key: str = "",
    chat_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Stream an answer word-by-word using Groq's cloud API."""
    if not api_key:
        yield "⚠️ Groq API key is required. Get a free key at https://console.groq.com"
        return

    prompt_content = build_prompt(question, context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Inject previous chat history for conversational context
    if chat_history:
        # Take the last few turns to avoid blowing up the context window
        recent_history = chat_history[-6:]
        for msg in recent_history:
            # We don't want to include the raw retrieved chunks from past assistant messages
            # to save tokens, just the text response.
            role = msg.get("role")
            content = msg.get("content", "")
            if role and content and not content.startswith("⚠️"):
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt_content})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1536,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, stream=True, timeout=60)
        if response.status_code == 429:
            yield "⏱️ Groq rate limit reached. Please wait a few seconds and try again!"
            return
        elif response.status_code != 200:
            yield f"⚠️ Groq API error (HTTP {response.status_code}): {response.text}"
            return

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        delta = chunk_json["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        pass
    except Exception as e:
        yield f"⚠️ Error generating stream: {str(e)}"


def stream_answer(
    question: str,
    context_chunks: list[str],
    model: str = "",
    api_key: str = "",
    chat_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Stream answer generator for Streamlit UI."""
    effective_model = model or DEFAULT_GROQ_MODEL
    return stream_answer_groq(question, context_chunks, effective_model, api_key, chat_history)


# ---------------------------------------------------------------------------
# Provider health checks
# ---------------------------------------------------------------------------


def check_groq_status(api_key: str) -> tuple[bool, str]:
    """Check if the Groq API is reachable and the key is valid.

    Args:
        api_key: The Groq API key to validate.

    Returns:
        A tuple of (is_ok, error_message). error_message is empty if ok.
    """
    if not api_key:
        return False, (
            "No Groq API key provided. Get a free key at "
            "https://console.groq.com (no credit card required)."
        )

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if response.status_code == 401:
            return False, "Invalid Groq API key. Check your key at https://console.groq.com"
        elif response.status_code == 200:
            return True, ""
        else:
            return False, f"Groq API returned status {response.status_code}"
    except requests.ConnectionError:
        return False, "Could not connect to Groq API. Check your internet connection."
    except requests.Timeout:
        return False, "Groq API connection timed out."
