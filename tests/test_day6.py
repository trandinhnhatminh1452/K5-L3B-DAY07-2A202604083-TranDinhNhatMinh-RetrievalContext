"""Compatibility tests for reusing actual Day 6 exports, without services."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from day6 import Day6Retriever, convert_chunks


def exported(text="scientific management", page=1):
    return {"chunk_id": "paragraph-export-id", "document_id": "book",
            "source": "book.html", "page": page, "text": text,
            "metadata": {"source_id": "book", "section": "introduction"}}


class Day6Tests(unittest.TestCase):
    def test_ids_match_exact_day6_upsert_formula(self):
        row = exported()
        expected = hashlib.sha256(json.dumps(
            ["C", "book", 1, row["text"], 0], ensure_ascii=False).encode()).hexdigest()
        chunk = convert_chunks([row], "C")[0]
        self.assertEqual(chunk["id"], expected)
        self.assertEqual(chunk["source"], "book.html")
        self.assertEqual(chunk["source_id"], "book")
        self.assertEqual(chunk["metadata"]["day6_export_chunk_id"], row["chunk_id"])

    def test_duplicate_text_has_distinct_occurrence_id(self):
        chunks = convert_chunks([exported(), exported()], "C")
        self.assertNotEqual(chunks[0]["id"], chunks[1]["id"])
        self.assertNotEqual(chunks[0]["id"], convert_chunks([exported()], "B")[0]["id"])

    def test_source_and_page_follow_store_metadata_precedence(self):
        row = exported()
        row["metadata"].update(source="db-source", page_number=7)
        chunk = convert_chunks([row], "C")[0]
        expected = hashlib.sha256(json.dumps(
            ["C", "db-source", 7, row["text"], 0], ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(chunk["id"], expected)
        self.assertEqual(chunk["page"], 7)

    def retriever(self, method):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        chunks = convert_chunks([exported(), exported("worker productivity", 2)], "C")
        (path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
        return Day6Retriever(path, method=method)

    def test_lexical_works_without_database_or_model(self):
        retriever = self.retriever("tfidf")
        with patch.object(retriever, "_connect", side_effect=AssertionError("database access")):
            hits = retriever.search("scientific", top_k=1)
        self.assertEqual(hits[0]["page"], 1)
        self.assertEqual(retriever.search("scientific", filters={"page": 2}), [])

    def test_invalid_filter_and_top_k_fail(self):
        retriever = self.retriever("tfidf")
        with self.assertRaises(ValueError):
            retriever.search("scientific", filters={"gold_page": 1})
        with self.assertRaises(ValueError):
            retriever.search("scientific", top_k=True)

    def test_hybrid_combines_ranks_not_incompatible_raw_scores(self):
        retriever = self.retriever("hybrid")
        first, second = retriever.chunks
        with patch.object(retriever, "_dense", return_value=[second, first]), patch.object(
                retriever, "_tfidf", return_value=[first]):
            hits = retriever.search("management", top_k=2)
        self.assertEqual(hits[0]["id"], first["id"])
        self.assertAlmostEqual(hits[0]["score"], 1 / 62 + 1 / 61)

    def test_dense_error_is_not_replaced_by_lexical_results(self):
        retriever = self.retriever("dense")
        with patch.object(retriever, "_embed", side_effect=RuntimeError("missing model")):
            with self.assertRaisesRegex(RuntimeError, "missing model"):
                retriever.search("scientific")


if __name__ == "__main__":
    unittest.main()
