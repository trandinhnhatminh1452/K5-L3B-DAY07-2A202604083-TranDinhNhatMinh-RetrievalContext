"""Reproducible Day07 experiments. Run `python src/day07.py --help`."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from agentic import run_agent
from agent_metrics import aggregate
from day6 import Day6Retriever, DEFAULT_DATA
from evidence import evidence_grade, fingerprint, read_json, strict_hit, validate_cases, write_json
from ollama_controller import OllamaController, SYSTEM_PROMPT

BENCHMARK = ROOT / "benchmark"
RESULTS = ROOT / "results"
CONFIGS = {"B0": ("dense", 800), "E1": ("dense", 4000),
           "M1": ("tfidf", 4000), "E2": ("hybrid", 4000)}
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512}


def now():
    return datetime.now(timezone.utc).isoformat()


def code_hash():
    files = sorted((ROOT / "src").glob("*.py")) + [ROOT / "starter" / n for n in
               ("agentic.py", "agent_metrics.py", "baseline.py", "day6.py", "convert_day6.py")]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): fingerprint(p) for p in files},
                                    sort_keys=True).encode()).hexdigest()


def absent_cases(split):
    items = {
        "dev": [
            ("What percentage of U.S. manufacturing firms used Taylor's scientific management in 2025?",
             "Full 283-chunk scan finds no 2025/adoption-rate survey; historical examples cannot establish a 2025 national percentage."),
            ("What fixed dollar cash premium does Taylor prescribe for every accepted worker improvement?",
             "All premium/reward passages checked, including page 41: a cash premium is required but no universal fixed dollar amount is specified. Task-specific bonuses are not universal innovation awards."),
        ],
        "test": [
            ("How many factories in Vietnam adopted Taylor's scientific management during 2024?",
             "Full 283-chunk scan finds no 2024, Vietnam or Vietnamese occurrence; no national adoption count is supplied."),
            ("What ISO certification number and certification renewal interval does Taylor require for scientific management?",
             "Full 283-chunk scan finds no standalone ISO or certification/audit specification; no certification number or renewal interval is supplied."),
        ],
    }
    return [{"id": f"ua-{split}-{i:02}", "split": split, "question": question,
             "answerable": False, "expected_answer": "Insufficient evidence in the collection.",
             "expected_behavior": "search_and_abstain", "gold": [], "absence_reason": reason}
            for i, (question, reason) in enumerate(items[split], 1)]


def prepare():
    if (BENCHMARK / "freeze.json").exists():
        verify_freeze()
        return {"status": "already_frozen", "freeze_sha256": fingerprint(BENCHMARK / "freeze.json")}
    chunks = read_json(DEFAULT_DATA / "chunks.json")
    protocol = {"configs": CONFIGS, "top_k": 3, "max_searches": 4, "filters": None,
                "context": "whole blocks, arrival order, ID deduplication; serialized characters",
                "options": OPTIONS, "selection": ["correct/all", "complete_context/answerable",
                "original_strict_hit3/answerable", "median_agent_ms_ascending", "B0,E1,M1,E2"],
                "errors_in_correctness_denominator": True,
                "test_reuse": "Five answerable questions already evaluated in Day06; user confirms no test-based tuning."}
    write_json(BENCHMARK / "protocol.json", protocol)
    for split in ("dev", "test"):
        cases = read_json(DEFAULT_DATA / f"questions_{split}.json")
        for q in cases:
            q["answerable"] = True
            if split == "dev" and q["id"] == "q13":
                texts = [
                    "And whenever the new method is found to be markedly superior to the old, it should be adopted as the standard for the whole establishment.",
                    "The workman should be given the full credit for the improvement, and should be paid a cash premium as a reward for his ingenuity."]
                original = q["gold"][0]
                if " ".join(texts) != original["evidence"]:
                    raise ValueError("q13 original evidence changed; review annotations explicitly")
                parts = []
                for text in texts:
                    matching = [c for c in chunks if c["source_id"] == original["source_id"]
                                and c["page"] == original["page"] and text in c["text"]]
                    if len(matching) != 1:
                        raise ValueError("q13 fragment mapping is not unique")
                    parts.append({**original, "evidence": text, "chunk_id": matching[0]["id"]})
                q["evidence_groups"] = [parts]
                q["annotation_note"] = "Original gold preserved; all fragments required, not OR alternatives."
        cases.extend(absent_cases(split))
        validate_cases(chunks, cases, split)
        write_json(BENCHMARK / f"{split}.json", cases)
    paths = [DEFAULT_DATA / "chunks.json", DEFAULT_DATA / "manifest.json",
             BENCHMARK / "dev.json", BENCHMARK / "test.json", BENCHMARK / "protocol.json"]
    freeze = {"created_utc": now(), "files": {str(p.relative_to(ROOT)): fingerprint(p) for p in paths},
              "counts": {s: dict(Counter("answerable" if q["answerable"] else "unanswerable"
                         for q in read_json(BENCHMARK / f"{s}.json"))) for s in ("dev", "test")}}
    write_json(BENCHMARK / "freeze.json", freeze)
    return freeze


def verify_freeze():
    freeze = read_json(BENCHMARK / "freeze.json")
    for path, expected in freeze["files"].items():
        if fingerprint(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}. Do not silently recreate the test lock.")
    return fingerprint(BENCHMARK / "freeze.json")


def check_runtime(model, base_url, timeout):
    controller = OllamaController(model, base_url=base_url, timeout=timeout)
    version = controller.request("/api/version")
    models = controller.request("/api/tags")["models"]
    match = [m for m in models if m.get("name") == model or m.get("model") == model]
    if len(match) != 1:
        raise ValueError(f"Model {model} is not available locally; no automatic download")
    return {"ollama_version": version, "model": model, "model_digest": match[0].get("digest")}


def verify_selection():
    selected = read_json(RESULTS / "selection.json")
    if selected["freeze_sha256"] != verify_freeze() or selected["code_sha256"] != code_hash():
        raise ValueError("Inputs or pipeline changed after dev selection")
    for path, sha in selected["artifacts_sha256"].items():
        if fingerprint(ROOT / path) != sha:
            raise ValueError("Dev selection artifact changed")
    return selected


def run(split, configs, model, base_url, timeout, resume=False, retrieval_only=False):
    freeze_sha = verify_freeze()
    if split == "test":
        selected = verify_selection()
        if retrieval_only or model != selected["model"] or set(configs) != {"B0", selected["config"]}:
            raise ValueError("Test must run baseline+selected with the selected model")
    runtime = None if retrieval_only else check_runtime(model, base_url, timeout)
    if split == "test" and runtime["model_digest"] != selected["model_digest"]:
        raise ValueError("Local model digest changed after dev selection")
    cases = read_json(BENCHMARK / f"{split}.json")
    chunks = read_json(DEFAULT_DATA / "chunks.json")
    validate_cases(chunks, cases, split)
    for config in configs:
        method, budget = CONFIGS[config]
        directory = RESULTS / ("retrieval_only" if retrieval_only else "agent") / split / config
        expected = {"split": split, "config": config, "method": method, "budget_chars": budget,
                    "model": None if retrieval_only else model, "runtime": runtime,
                    "freeze_sha256": freeze_sha, "code_sha256": code_hash(), "filters": None,
                    "top_k": 3, "max_searches": 4, "options": OPTIONS,
                    "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
        path = directory / "traces.json"
        traces = []
        if path.exists():
            if not resume or read_json(directory / "run.json")["parameters"] != expected:
                raise ValueError(f"Output exists or parameters changed: {directory}; choose matching --resume")
            traces = read_json(path)
            if [t["query_id"] for t in traces] != [q["id"] for q in cases[:len(traces)]]:
                raise ValueError("Resume IDs do not match case order")
        setup = time.perf_counter()
        retriever = Day6Retriever(method=method)
        verification = retriever.check(verify_embedding=method != "tfidf")
        write_json(directory / "run.json", {"parameters": expected, "verified": verification,
                   "started_utc": now(), "setup_seconds": time.perf_counter() - setup})
        for q in cases[len(traces):]:
            start = time.perf_counter()
            record = {"query_id": q["id"], "split": split, "config": config,
                      "question": q["question"], "answerable": q["answerable"],
                      "filter": None, "budget_chars": budget, "events": [], "citations": []}
            controller = OllamaController(model, base_url=base_url, timeout=timeout)
            try:
                raw = retriever.search(q["question"], filters=None, top_k=3)
                record.update(original_results=raw, original_ranking_ms=(time.perf_counter()-start)*1000,
                              hit1=strict_hit(raw[:1], q), hit3=strict_hit(raw, q),
                              collective_hit3=evidence_grade(raw, q) == 2 if q["answerable"] else None)
                agent_start = time.perf_counter()
                if retrieval_only:
                    from baseline import pack_context
                    context, ids = pack_context(raw, budget)
                    record.update(status="retrieval_only", answer=None,
                                  events=[{"event": "search", "query": q["question"],
                                           "results": raw, "context": context, "context_ids": ids}])
                else:
                    # Only question reaches the agent: gold stays exclusively in evaluator.
                    record.update(run_agent(q["question"], controller, retriever,
                                  max_searches=4, top_k=3, budget=budget, event_log=record["events"]))
                record["agent_elapsed_ms"] = (time.perf_counter() - agent_start) * 1000
                searches = [e for e in record["events"] if e["event"] == "search"]
                last = searches[-1] if searches else {}
                record["context"] = last.get("context", "")
                record["context_ids"] = last.get("context_ids", [])
                chosen = [c for c in chunks if c["id"] in record["context_ids"]]
                record["context_evidence_grade"] = evidence_grade(chosen, q)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            # Preserve the last real context and successful tool calls even if a later model call fails.
            searches = [e for e in record["events"] if e["event"] == "search"]
            last = searches[-1] if searches else {}
            record["context"] = last.get("context", "")
            record["context_ids"] = last.get("context_ids", [])
            chosen = [c for c in chunks if c["id"] in record["context_ids"]]
            record["context_evidence_grade"] = evidence_grade(chosen, q)
            record.update(model_calls=controller.calls, usage=controller.usage(),
                          total_elapsed_ms=(time.perf_counter()-start)*1000)
            traces.append(record)
            write_json(path, traces)
            print(f"{split}/{config}/{q['id']}: {record['status']}", flush=True)
        write_json(directory / "retrieval_summary.json", retrieval_summary(cases, traces))
    return {"status": "finished", "split": split, "configs": configs, "retrieval_only": retrieval_only}


def ratio(n, total):
    return {"n": n, "N": total, "rate": n / total if total else None}


def retrieval_summary(cases, traces):
    answerable_ids = {q["id"] for q in cases if q["answerable"]}
    valid = [t for t in traces if t["query_id"] in answerable_ids]
    n = len(answerable_ids)
    return {"cases": len(cases), "errors": sum(t["status"] == "error" for t in traces),
            "hit1": ratio(sum(bool(t.get("hit1")) for t in valid), n),
            "hit3": ratio(sum(bool(t.get("hit3")) for t in valid), n),
            "collective_hit3": ratio(sum(bool(t.get("collective_hit3")) for t in valid), n),
            "context_evidence": ratio(sum(t.get("context_evidence_grade") == 2 for t in valid), n),
            "median_original_ranking_ms": statistics.median(t["original_ranking_ms"] for t in traces
                            if "original_ranking_ms" in t) if any("original_ranking_ms" in t for t in traces) else None}


def review_packet(split, config):
    cases = read_json(BENCHMARK / f"{split}.json")
    directory = RESULTS / "agent" / split / config
    traces = read_json(directory / "traces.json")
    by_id = {t["query_id"]: t for t in traces}
    packet, reviews = [], []
    for q in cases:
        t = by_id[q["id"]]
        packet.append({"case": q, "trace": t})
        reviews.append({"query_id": q["id"], "answer_grade": None, "context_grade": None,
                        "abstention_correct": None, "citation_support": [None] * len(t["citations"]),
                        "citation_reasons": [""] * len(t["citations"]), "reason": ""})
    write_json(directory / "audit_packet.json", packet)
    template = directory / "reviews.template.json"
    if not template.exists():
        write_json(template, {"reviewer": "", "student_verified": False, "reviews": reviews})
    return {"packet": str(directory / "audit_packet.json"), "template": str(template)}


def summarize(split, config, reviews_path):
    verify_freeze()
    cases = read_json(BENCHMARK / f"{split}.json")
    directory = RESULTS / "agent" / split / config
    traces = read_json(directory / "traces.json")
    run_info = read_json(directory / "run.json")["parameters"]
    if run_info["code_sha256"] != code_hash() or run_info["freeze_sha256"] != verify_freeze():
        raise ValueError("Run pipeline/input hashes no longer match")
    review_doc = read_json(reviews_path)
    if not review_doc.get("reviewer") or type(review_doc.get("student_verified")) is not bool:
        raise ValueError("Identify reviewer and student-verification status")
    reviews = review_doc["reviews"]
    review_map = {r["query_id"]: r for r in reviews}
    trace_map = {t["query_id"]: t for t in traces}
    correct = 0
    for q in cases:
        t = trace_map[q["id"]]
        if t["status"] == "error":
            continue
        r = review_map[q["id"]]
        if not r.get("reason", "").strip():
            raise ValueError(f"{q['id']}: review needs a source-based rationale")
        if q["answerable"] and (type(r.get("context_grade")) is not int or r["context_grade"] not in (0, 1, 2)):
            raise ValueError(f"{q['id']}: context grade must be 0/1/2")
        if len(r.get("citation_reasons", [])) != len(t["citations"]) or any(not x.strip() for x in r["citation_reasons"]):
            raise ValueError(f"{q['id']}: each citation needs a source-support reason")
        for cid, supported in zip(t["citations"], r["citation_support"]):
            if supported and (cid not in t.get("context_ids", []) or f"[{cid}]" not in t.get("answer", "")):
                raise ValueError("A supported citation must also be in final context and answer")
        if q["answerable"]:
            correct += (t["status"] == "answered" and r["answer_grade"] == 2
                        and bool(t["citations"]) and all(r["citation_support"]))
        else:
            correct += bool(r["abstention_correct"] and t["status"] == "abstain")
    metrics = aggregate(cases, traces, reviews)
    retrieval = retrieval_summary(cases, traces)
    histogram = Counter(sum(e["event"] == "search" for e in t["events"]) for t in traces if t["status"] != "error")
    latencies = [t["agent_elapsed_ms"] for t in traces if "agent_elapsed_ms" in t]
    summary = {"config": config, "split": split, "model": run_info["model"],
               "model_digest": run_info["runtime"]["model_digest"],
               "freeze_sha256": run_info["freeze_sha256"], "code_sha256": run_info["code_sha256"],
               "correct": ratio(correct, len(cases)), "retrieval": retrieval, "agent": metrics,
               "context_grades": dict(Counter(str(r["context_grade"]) for r in reviews if r.get("context_grade") is not None)),
               "retrieval_calls_distribution": dict(histogram),
               "median_agent_ms": statistics.median(latencies) if latencies else None,
               "usage": {k: sum(t["usage"][k] for t in traces) for k in
                         ("model_calls", "prompt_tokens", "completion_tokens", "service_cost_usd")},
               "cost_note": "Local service charges 0 USD; hardware/electricity unmeasured",
               "reviewer": review_doc["reviewer"], "student_verified": review_doc["student_verified"],
               "reviews_sha256": fingerprint(reviews_path), "traces_sha256": fingerprint(directory / "traces.json")}
    write_json(directory / "reviews.json", review_doc)
    write_json(directory / "summary.json", summary)
    return summary


def select():
    verify_freeze()
    summaries = [read_json(RESULTS / "agent/dev" / c / "summary.json") for c in CONFIGS]
    contracts = {(s["model"], s["model_digest"], s["freeze_sha256"], s["code_sha256"]) for s in summaries}
    if (len(contracts) != 1 or summaries[0]["code_sha256"] != code_hash()
            or summaries[0]["freeze_sha256"] != verify_freeze()):
        raise ValueError("Dev comparisons must share model, inputs and pipeline")
    def key(s):
        return (-s["correct"]["rate"], -s["retrieval"]["context_evidence"]["rate"],
                -s["retrieval"]["hit3"]["rate"], s["median_agent_ms"] if s["median_agent_ms"] is not None else float("inf"),
                list(CONFIGS).index(s["config"]))
    best = min(summaries, key=key)
    paths = [RESULTS / "agent/dev" / c / n for c in CONFIGS for n in ("summary.json", "reviews.json", "traces.json", "run.json")]
    for c, s in zip(CONFIGS, summaries):
        directory = RESULTS / "agent/dev" / c
        if s["reviews_sha256"] != fingerprint(directory / "reviews.json") or s["traces_sha256"] != fingerprint(directory / "traces.json"):
            raise ValueError("Dev reviews/traces changed after summarization")
    selection = {"selected_utc": now(), "config": best["config"], "model": best["model"],
                 "model_digest": best["model_digest"], "freeze_sha256": verify_freeze(), "code_sha256": code_hash(),
                 "ranking": [s["config"] for s in sorted(summaries, key=key)],
                 "artifacts_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in paths}}
    if (RESULTS / "selection.json").exists():
        raise ValueError("Selection already exists; do not silently reselect after test")
    write_json(RESULTS / "selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    check = sub.add_parser("check")
    run_cmd = sub.add_parser("run")
    for cmd in (check, run_cmd):
        cmd.add_argument("--model", default="llama3.2:3b")
        cmd.add_argument("--base-url", default="http://127.0.0.1:11434")
        cmd.add_argument("--timeout", type=float, default=180)
    run_cmd.add_argument("--split", choices=("dev", "test"), required=True)
    run_cmd.add_argument("--configs", choices=tuple(CONFIGS), nargs="+", default=list(CONFIGS))
    run_cmd.add_argument("--resume", action="store_true")
    run_cmd.add_argument("--retrieval-only", action="store_true", help="Diagnostic dev retrieval, never an agent benchmark")
    for name in ("review-packet", "summarize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--split", choices=("dev", "test"), required=True)
        cmd.add_argument("--config", choices=tuple(CONFIGS), required=True)
        if name == "summarize":
            cmd.add_argument("--reviews", type=Path, required=True)
    sub.add_parser("select")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "check":
        result = check_runtime(args.model, args.base_url, args.timeout)
    elif args.command == "run":
        result = run(args.split, list(dict.fromkeys(args.configs)), args.model, args.base_url,
                     args.timeout, args.resume, args.retrieval_only)
    elif args.command == "review-packet":
        result = review_packet(args.split, args.config)
    elif args.command == "summarize":
        result = summarize(args.split, args.config, args.reviews)
    else:
        result = select()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
