"""Reuse this student's Day 6 exports and pgvector index without re-ingestion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from baseline import TfidfIndex
from convert_day6 import convert_question

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "starter" / "data" / "day6"


def find_day6(root=None):
    if root is not None:
        candidate = Path(root).expanduser().resolve()
        if (candidate / "starter" / "results" / "run_metadata.json").is_file():
            return candidate
        raise FileNotFoundError(f"Not a Day 6 repository: {candidate}")
    candidates = [p for p in ROOT.parent.glob("*DAY06*")
                  if (p / "starter/results/run_metadata.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("Pass --day6-root: expected exactly one sibling Day 6 repository")
    return candidates[0].resolve()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def convert_chunks(rows, strategy):
    """Reproduce PgVectorStore.upsert IDs; preserve export IDs for audit."""
    occurrences = Counter()
    chunks = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        db_source = str(metadata.get("source") or metadata.get("source_id") or row["source"])
        page = int(metadata.get("page", metadata.get("page_number", row["page"])))
        key = (db_source, page, row["text"])
        raw_id = json.dumps([strategy, db_source, page, row["text"], occurrences[key]],
                            ensure_ascii=False)
        occurrences[key] += 1
        chunks.append({
            "id": hashlib.sha256(raw_id.encode("utf-8")).hexdigest(),
            "source_id": metadata.get("source_id", row["document_id"]),
            "source": row["source"], "page": page, "text": row["text"],
            "metadata": {**metadata, "source": db_source, "page": page,
                         "day6_export_chunk_id": row["chunk_id"],
                         "day6_strategy": strategy},
        })
    if not chunks or len({c["id"] for c in chunks}) != len(chunks):
        raise ValueError("Empty chunk export or duplicate IDs")
    return chunks


def prepare(day6_root, strategy="C", output=DEFAULT_DATA):
    day6_root = find_day6(day6_root)
    results = day6_root / "starter/results"
    run = json.loads((results / "run_metadata.json").read_text(encoding="utf-8-sig"))
    model = run["embedding_model"]
    if (model["model_name"] != "intfloat/multilingual-e5-small"
            or model["dimensions"] != 384
            or model["query_preprocessing"] != "'query: ' + question text"
            or model["document_preprocessing"] != "'passage: ' + chunk text"
            or model["normalize_embeddings"] is not True
            or not model.get("revision")):
        raise ValueError("Unexpected Day 6 embedding contract; update the adapter explicitly")
    chunk_path = results / f"chunks_{strategy}.jsonl"
    chunks = convert_chunks(read_jsonl(chunk_path), strategy)
    if len(chunks) != run["strategies"][strategy]["chunk_count"]:
        raise ValueError("Chunk export count does not match run metadata")
    output = Path(output)
    payloads = {"chunks.json": chunks}
    counts = {}
    fingerprints = {}
    for split in ("dev", "test"):
        path = day6_root / "benchmark" / f"{split}.jsonl"
        questions = []
        for row in read_jsonl(path):
            if row.get("split", split) != split:
                raise ValueError(f"Wrong split in {path}")
            questions.append(convert_question({**row, "split": split}))
        payloads[f"questions_{split}.json"] = questions
        counts[split] = len(questions)
        fingerprints[f"benchmark/{split}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "day6_root": str(day6_root), "strategy": strategy,
        "chunk_count": len(chunks), "question_counts": counts,
        "embedding_model": model, "query_prefix": "query: ",
        "vector_store": run["vector_store"], "table": "rag_chunks",
        "day6_files_sha256": {
            **fingerprints,
            f"starter/results/chunks_{strategy}.jsonl": hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
            "starter/results/run_metadata.json": hashlib.sha256((results / "run_metadata.json").read_bytes()).hexdigest(),
        },
        "database_verified": False,
        "notes": [
            "IDs reconstructed with the exact Day 6 store hash; run check to verify against PostgreSQL.",
            "page is a logical HTML unit, not a printed page.",
            "Day 6 benchmark contains 20 answerable cases; add 2 unanswerable per split for Day 7.",
            "Test is converted only; never used to tune retrieval.",
            "Gold evidence can cross paragraph chunks (e.g. q13); preserve gold and evaluate multi-chunk evidence separately.",
        ],
    }
    payloads["manifest.json"] = manifest
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    return manifest


class Day6Retriever:
    """Read-only dense / TF-IDF / RRF retrieval on the same Day 6 strategy."""

    def __init__(self, data_dir=DEFAULT_DATA, *, method="dense", allow_download=False):
        if method not in ("dense", "tfidf", "hybrid"):
            raise ValueError("method must be dense, tfidf or hybrid")
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = json.loads((self.data_dir / "chunks.json").read_text(encoding="utf-8"))
        self.method = method
        self.allow_download = allow_download
        self._model = None
        self._lexical = None
        self._database_url = None

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row
        if self._database_url is None:
            from dotenv import dotenv_values
            path = Path(self.manifest["day6_root"]) / "starter/.env"
            self._database_url = os.environ.get("DATABASE_URL") or dotenv_values(path).get("DATABASE_URL")
        if not self._database_url:
            raise ValueError("Set DATABASE_URL or use Day 6 starter/.env")
        # A server-side read-only transaction protects the existing Day 6 index.
        return psycopg.connect(self._database_url, row_factory=dict_row,
                               connect_timeout=5, options="-c default_transaction_read_only=on -c statement_timeout=15000")

    def check(self, *, verify_embedding=False):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source, page, text, metadata, vector_dims(embedding) AS dimensions "
                "FROM rag_chunks WHERE strategy = %s", (self.manifest["strategy"],)
            ).fetchall()
        actual = {row["id"]: row for row in rows}
        if len(actual) != len(self.chunks) or set(actual) != {c["id"] for c in self.chunks}:
            raise ValueError("Database IDs/count differ from Day 6 export; do not rebuild blindly")
        for chunk in self.chunks:
            row = actual[chunk["id"]]
            if (row["text"] != chunk["text"] or row["page"] != chunk["page"]
                    or row["source"] != chunk["metadata"]["source"]
                    or row["dimensions"] != self.manifest["embedding_model"]["dimensions"]
                    or any(row["metadata"].get(k) != v for k, v in chunk["metadata"].items()
                           if k not in ("day6_export_chunk_id", "day6_strategy"))):
                raise ValueError(f"Database content/metadata/dimensions mismatch: {chunk['id']}")
        result = {"status": "ok", "strategy": self.manifest["strategy"],
                "verified_chunks": len(rows),
                "dimensions": self.manifest["embedding_model"]["dimensions"],
                "embedding_provenance": "Day 6 run metadata; database does not store model identity",
                "database_mode": "read-only"}
        if verify_embedding:
            sample = self.chunks[0]
            vector = "[" + ",".join(map(str, self._encode("passage: " + sample["text"]))) + "]"
            with self._connect() as conn:
                score = conn.execute(
                    "SELECT 1 - (embedding <=> %s::vector) AS score "
                    "FROM rag_chunks WHERE id = %s", (vector, sample["id"])
                ).fetchone()["score"]
            if score is None or float(score) < 0.999:
                raise ValueError("Sample document vector does not match the pinned Day 6 embedding model")
            result["embedding_sample"] = {"id": sample["id"], "cosine_similarity": float(score)}
        return result

    def _embed(self, query):
        return self._encode(self.manifest["query_prefix"] + query)

    def _encode(self, text):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            info = self.manifest["embedding_model"]
            old_cache = Path(self.manifest["day6_root"]) / "starter/.cache/huggingface"
            cache = old_cache if old_cache.is_dir() else ROOT / ".cache/huggingface"
            self._model = SentenceTransformer(
                info["model_name"], revision=info["revision"], device="cpu",
                cache_folder=str(cache), local_files_only=not self.allow_download,
                trust_remote_code=False)
            if self._model.get_sentence_embedding_dimension() != info["dimensions"]:
                raise ValueError("Loaded model dimensions differ from the Day 6 index")
        if len(self._model.tokenizer(text, truncation=False)["input_ids"]) > self._model.max_seq_length:
            raise ValueError("Input exceeds the embedding model token limit")
        vector = self._model.encode([text], normalize_embeddings=True, precision="float32",
                                    prompt="", show_progress_bar=False)[0].tolist()
        if (len(vector) != self.manifest["embedding_model"]["dimensions"]
                or not all(math.isfinite(v) for v in vector)
                or abs(sum(v * v for v in vector) - 1) > 0.002):
            raise ValueError("Invalid embedding")
        return vector

    @staticmethod
    def _matches(chunk, filters):
        allowed = {"source", "source_id", "page", "section", "language", "title"}
        if not isinstance(filters, (dict, type(None))):
            raise ValueError("filters must be an object")
        if set(filters or {}) - allowed:
            raise ValueError("Unsupported filter key")
        for key, value in (filters or {}).items():
            if key == "source":
                actual = chunk["metadata"]["source"]
            else:
                actual = chunk.get(key, chunk["metadata"].get(key))
            if actual != value:
                return False
        return True

    def _dense(self, query, filters, top_k):
        vector = "[" + ",".join(map(str, self._embed(query))) + "]"
        # Filter on exported metadata before ranking with the real database vectors.
        ids = [c["id"] for c in self.chunks if self._matches(c, filters)]
        if not ids:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, 1 - (embedding <=> %s::vector) AS score FROM rag_chunks "
                "WHERE strategy = %s AND id = ANY(%s::text[]) "
                "ORDER BY embedding <=> %s::vector, id LIMIT %s",
                (vector, self.manifest["strategy"], ids, vector, top_k)).fetchall()
        chunks = {c["id"]: c for c in self.chunks}
        return [{**chunks[r["id"]], "score": float(r["score"]), "strategy": "dense"}
                for r in rows]

    def _tfidf(self, query, filters, top_k):
        if self._lexical is None:
            self._lexical = TfidfIndex(self.chunks)
        # Fit once on the full corpus; filter candidates before selecting top-k.
        hits = self._lexical.search(query, top_k=len(self.chunks))
        return [c for c in hits if self._matches(c, filters)][:top_k]

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip() or type(top_k) is not int or top_k < 1:
            raise ValueError("Pass a nonempty query and positive integer top_k")
        # Validate filter keys even when the corpus has no matching candidates.
        self._matches(self.chunks[0], filters)
        if self.method == "dense":
            return self._dense(query, filters, top_k)
        if self.method == "tfidf":
            return self._tfidf(query, filters, top_k)
        scores, chunks = Counter(), {}
        for ranking in (self._dense(query, filters, max(20, top_k)),
                        self._tfidf(query, filters, max(20, top_k))):
            for rank, chunk in enumerate(ranking, 1):
                scores[chunk["id"]] += 1 / (60 + rank)
                chunks[chunk["id"]] = chunk
        return [{**chunks[id], "score": scores[id], "strategy": "hybrid_rrf"}
                for id in sorted(scores, key=lambda id: (-scores[id], id))[:top_k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Convert local files; no embedding/database writes")
    prep.add_argument("--day6-root", type=Path)
    prep.add_argument("--strategy", choices=("A", "B", "C"), default="C")
    prep.add_argument("--output", type=Path, default=DEFAULT_DATA)
    for name in ("check", "search"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data", type=Path, default=DEFAULT_DATA)
        cmd.add_argument("--allow-download", action="store_true")
        if name == "check":
            cmd.add_argument("--with-model", action="store_true")
        if name == "search":
            cmd.add_argument("--query", required=True)
            cmd.add_argument("--method", choices=("dense", "tfidf", "hybrid"), default="dense")
            cmd.add_argument("--top-k", type=int, default=3)
            cmd.add_argument("--filters", type=json.loads)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.day6_root, args.strategy, args.output)
        result = {k: result[k] for k in ("strategy", "chunk_count", "question_counts", "database_verified")}
    else:
        retriever = Day6Retriever(args.data, method=getattr(args, "method", "dense"),
                                  allow_download=getattr(args, "allow_download", False))
        if args.command == "check":
            result = retriever.check(verify_embedding=args.with_model)
        else:
            if args.method != "tfidf":
                retriever.check()
            result = retriever.search(args.query, filters=args.filters, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
