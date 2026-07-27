"""
test_rag_pipeline.py — Unit tests for the RAG pipeline.

Tests chunking logic (pure functions, no external dependencies) and
retrieval/prompt logic (mocked embedder + collection).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_prompt,
    chunk_text,
    extract_text_from_pdf,
    query_documents,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Chunking tests — pure functions, zero external dependencies
# ---------------------------------------------------------------------------


class TestChunkText:
    """Tests for the chunk_text() function."""

    def test_chunk_empty_string(self) -> None:
        """Empty input should return an empty list."""
        assert chunk_text("") == []

    def test_chunk_whitespace_only(self) -> None:
        """Whitespace-only input should return an empty list."""
        assert chunk_text("   \n\n  \t  ") == []

    def test_chunk_text_shorter_than_chunk_size(self) -> None:
        """Text shorter than chunk_size should return a single chunk."""
        short_text = "This is a short text."
        result = chunk_text(short_text, chunk_size=800, overlap=150)
        assert len(result) == 1
        assert result[0] == short_text

    def test_chunk_text_exactly_chunk_size(self) -> None:
        """Text exactly equal to chunk_size should return a single chunk."""
        text = "a" * CHUNK_SIZE
        result = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_produces_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks."""
        # Create text that's clearly longer than one chunk
        text = "This is a sentence. " * 200  # ~4000 chars
        result = chunk_text(text, chunk_size=800, overlap=150)
        assert len(result) > 1

    def test_chunk_overlap_present(self) -> None:
        """Adjacent chunks should share overlapping content."""
        # Create distinct sentences to track overlap
        sentences = [f"Sentence number {i} is here. " for i in range(100)]
        text = "".join(sentences)

        result = chunk_text(text, chunk_size=300, overlap=80)
        assert len(result) >= 2

        # Check that end of chunk N appears at the start of chunk N+1
        for i in range(len(result) - 1):
            # The tail of the current chunk should appear in the next chunk
            tail = result[i][-40:]  # Last 40 chars of current
            # At least some of the tail content should appear in the next chunk
            assert any(
                word in result[i + 1] for word in tail.split() if len(word) > 3
            ), f"No overlap detected between chunk {i} and {i + 1}"

    def test_chunk_no_mid_word_split(self) -> None:
        """Chunks should not split in the middle of a word."""
        text = "Supercalifragilistic " * 100  # Long word repeated
        result = chunk_text(text, chunk_size=200, overlap=50)

        for chunk in result:
            # Each chunk should start and end at word boundaries
            stripped = chunk.strip()
            if stripped:
                # Should not start with a partial word (lowercase continuation)
                # Actually, check that the chunk contains complete words
                words = stripped.split()
                for word in words:
                    assert word == "Supercalifragilistic" or word == "", (
                        f"Found partial word: '{word}'"
                    )

    def test_chunk_preserves_all_content(self) -> None:
        """All original content should be present across all chunks combined."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        words_original = set(text.split())

        result = chunk_text(text, chunk_size=200, overlap=50)
        words_chunked = set()
        for chunk in result:
            words_chunked.update(chunk.split())

        # All original words should appear in at least one chunk
        assert words_original.issubset(words_chunked)

    def test_chunk_respects_paragraph_boundaries(self) -> None:
        """Chunking should prefer paragraph boundaries when possible."""
        paragraph1 = "A" * 400
        paragraph2 = "B" * 400
        text = f"{paragraph1}\n\n{paragraph2}"

        result = chunk_text(text, chunk_size=500, overlap=50)
        # Should split at the paragraph boundary
        assert len(result) >= 2

    def test_chunk_with_zero_overlap(self) -> None:
        """Zero overlap should produce non-overlapping chunks."""
        text = "Word " * 200
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert len(result) > 1
        # Total content length should be close to original
        total_len = sum(len(c) for c in result)
        assert total_len >= len(text.strip()) * 0.8  # Allow some boundary loss

    def test_chunk_single_long_word(self) -> None:
        """A single very long word should still produce output."""
        text = "x" * 2000
        result = chunk_text(text, chunk_size=800, overlap=150)
        assert len(result) >= 1
        # All characters should be covered
        total = "".join(result)
        assert "x" * 800 in total


# ---------------------------------------------------------------------------
# PDF extraction tests
# ---------------------------------------------------------------------------


class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf()."""

    def test_extract_from_invalid_input(self) -> None:
        """Invalid input should return empty string, not crash."""
        import io

        result = extract_text_from_pdf(io.BytesIO(b"not a pdf"))
        assert result == ""

    def test_extract_from_empty_bytes(self) -> None:
        """Empty bytes should return empty string."""
        import io

        result = extract_text_from_pdf(io.BytesIO(b""))
        assert result == ""


# ---------------------------------------------------------------------------
# Prompt assembly tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for build_prompt()."""

    def test_build_prompt_includes_question(self) -> None:
        """The question should appear in the assembled prompt."""
        question = "What is machine learning?"
        chunks = ["Machine learning is a subset of AI."]
        result = build_prompt(question, chunks)
        assert question in result

    def test_build_prompt_includes_context(self) -> None:
        """All context chunks should appear in the prompt."""
        chunks = [
            "First chunk of text.",
            "Second chunk of text.",
            "Third chunk of text.",
        ]
        result = build_prompt("test question", chunks)
        for chunk in chunks:
            assert chunk in result

    def test_build_prompt_includes_source_labels(self) -> None:
        """Each chunk should be labeled with a source number."""
        chunks = ["Chunk A", "Chunk B"]
        result = build_prompt("question", chunks)
        assert "[Source 1]" in result
        assert "[Source 2]" in result

    def test_build_prompt_no_chunks(self) -> None:
        """Empty chunks should produce a 'no context' message."""
        result = build_prompt("question", [])
        assert "No relevant context" in result


# ---------------------------------------------------------------------------
# Retrieval tests (mocked)
# ---------------------------------------------------------------------------


class TestQueryDocuments:
    """Tests for query_documents() using mocked ChromaDB."""

    def test_query_returns_expected_structure(self) -> None:
        """Query results should have documents, metadatas, and distances keys."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test.pdf", "chunk_index": 0},
                           {"source": "test.pdf", "chunk_index": 1}]],
            "distances": [[0.1, 0.3]],
        }

        mock_embed_fn = MagicMock()
        mock_embed_fn.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[[0.1] * 384])
        )

        results = query_documents("test query", mock_collection, mock_embed_fn, top_k=2)

        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert len(results["documents"][0]) == 2

    def test_query_empty_collection(self) -> None:
        """Querying an empty collection should return empty structure."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        results = query_documents("test", mock_collection, top_k=4)

        assert results["documents"] == [[]]
        assert results["metadatas"] == [[]]
        assert results["distances"] == [[]]

    def test_query_top_k_capped_by_collection_size(self) -> None:
        """top_k should be capped to collection size when collection is small."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "a.pdf", "chunk_index": 0},
                           {"source": "b.pdf", "chunk_index": 0}]],
            "distances": [[0.2, 0.4]],
        }

        mock_embed_fn = MagicMock()
        mock_embed_fn.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[[0.1] * 384])
        )

        results = query_documents("test", mock_collection, mock_embed_fn, top_k=10)

        # Should have queried with n_results=2, not 10
        call_args = mock_collection.query.call_args
        assert call_args.kwargs["n_results"] == 2


# ---------------------------------------------------------------------------
# Integration-style prompt + retrieval test
# ---------------------------------------------------------------------------


class TestPromptWithRetrieval:
    """Tests verifying prompt assembly with retrieval results."""

    def test_full_prompt_flow(self) -> None:
        """Verify the complete flow from retrieval results to prompt."""
        # Simulate retrieval results
        chunks = [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]
        question = "Who created Python?"

        prompt = build_prompt(question, chunks)

        # Verify structure
        assert "Who created Python?" in prompt
        assert "Python is a programming language" in prompt
        assert "Guido van Rossum" in prompt
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        assert "CONTEXT" in prompt
