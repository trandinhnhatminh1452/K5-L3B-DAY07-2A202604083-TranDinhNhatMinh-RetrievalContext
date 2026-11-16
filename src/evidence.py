"""Evidence labels: OR across alternatives, AND within a required group."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches(chunk, evidence):
    return (chunk["source_id"] == evidence["source_id"]
            and chunk["page"] == evidence["page"]
            and evidence["evidence"] in chunk["text"]
            and (not evidence.get("chunk_id") or chunk["id"] == evidence["chunk_id"]))


def groups(case):
    return case.get("evidence_groups") or [[g] for g in case.get("gold", [])]


def strict_hit(chunks, case):
    return any(matches(c, g) for c in chunks for g in case.get("gold", []))


def evidence_grade(chunks, case):
    if not case["answerable"]:
        return None
    alternatives = groups(case)
    if any(all(any(matches(c, part) for c in chunks) for part in group)
           for group in alternatives):
        return 2
    return int(any(matches(c, part) for c in chunks
                   for group in alternatives for part in group))


def validate_cases(chunks, cases, split):
    if not chunks or not cases or split not in ("dev", "test"):
        raise ValueError("Nonempty corpus/cases and an explicit split are required")
    ids = set()
    for c in chunks:
        if not all(c.get(k) is not None for k in ("id", "text", "source_id", "source", "page", "metadata")):
            raise ValueError("Incomplete chunk")
        if c["id"] in ids or not isinstance(c["metadata"], dict):
            raise ValueError("Duplicate chunk ID or invalid metadata")
        ids.add(c["id"])
    ids = set()
    for q in cases:
        if not isinstance(q.get("id"), str) or not q["id"] or q["id"] in ids:
            raise ValueError("Invalid/duplicate case ID")
        ids.add(q["id"])
        if q.get("split") != split or not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError(f"{q['id']}: wrong split or empty question")
        if type(q.get("answerable")) is not bool:
            raise ValueError(f"{q['id']}: answerable must be boolean")
        if not q["answerable"]:
            if q.get("gold") or q.get("evidence_groups") or not q.get("absence_reason"):
                raise ValueError(f"{q['id']}: absent case needs a rationale and no gold")
            continue
        if not q.get("expected_answer") or not q.get("gold"):
            raise ValueError(f"{q['id']}: missing answer/gold")
        for group in groups(q):
            if not group or not all(any(matches(c, part) for c in chunks) for part in group):
                raise ValueError(f"{q['id']}: required fragment absent from corpus")
        for original in q["gold"]:
            if any(matches(c, original) for c in chunks):
                continue
            preserved = any(
                " ".join(p["evidence"] for p in group) == original["evidence"]
                and all(p["source_id"] == original["source_id"] and p["page"] == original["page"]
                        for p in group) for group in groups(q))
            if not preserved:
                raise ValueError(f"{q['id']}: original gold not preserved by required fragments")
