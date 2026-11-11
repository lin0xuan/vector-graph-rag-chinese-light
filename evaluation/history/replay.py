"""Recompute the historical metrics without any model, index or corpus."""
import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def read_latest(path):
    with path.open(encoding="utf-8") as stream:
        return {row["id"]: row for row in map(json.loads, stream)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def thousand(group):
    counts = Counter(row.get("accuracy") for row in group)
    totals = {
        name: {
            "hits": sum(row[key] for row in group),
            "gold": sum(row["count"] for row in group),
            "macro": mean(row[key] / row["count"] for row in group),
        }
        for name, key in (("candidate_coverage", "pool_hit"), ("recall_at_5", "top5_hit"),
                          ("evidence_coverage", "used_hit"))
    }
    return {
        "questions": len(group), "graded": sum(row.get("accuracy") is not None for row in group),
        "complete_correct": counts["correct"], "partial": counts["partial"], "wrong": counts["wrong"],
        "no_evidence": sum(row["status"] == "no_evidence" for row in group),
        "grounded": sum(row.get("grounded") is True for row in group),
        "fallbacks": sum(row["fallback"] for row in group),
        **totals,
        "candidate_all_sources": sum(row["pool_hit"] == row["count"] for row in group),
        "evidence_all_sources": sum(row["used_hit"] == row["count"] for row in group),
        "candidates_missing_sources": sum(row["pool_hit"] < row["count"] for row in group),
        "sources_lost_after_candidates": sum(row["pool_hit"] == row["count"] > row["used_hit"]
                                             for row in group),
    }


def transitions(rows, test):
    pairs = Counter((test(row, "base"), test(row, "new")) for row in rows)
    return {"base": pairs[True, True] + pairs[True, False],
            "new": pairs[True, True] + pairs[False, True],
            "improved": pairs[False, True], "regressed": pairs[True, False]}


def verify_raw(raw_dir, base, rerank, audit, answer):
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        require(hashlib.sha256((raw_dir / entry["file"]).read_bytes()).hexdigest() == entry["sha256"],
                f"Raw artifact checksum changed: {entry['file']}")
    generated = read_latest(raw_dir / "oa-qwen-1000-generated.jsonl")
    graded = read_latest(raw_dir / "oa-qwen-1000-graded.jsonl")
    frozen = read_latest(raw_dir / "oa-rerank-cover-frozen.jsonl")
    ab = read_latest(raw_dir / "oa-rerank-cover-ab.jsonl")
    audited = read_latest(raw_dir / "oa-rerank-audit-43.jsonl")
    answered = read_latest(raw_dir / "oa-answer-ab-32.jsonl")
    require(base.keys() == generated.keys() and graded.keys() <= base.keys(), "Raw thousand IDs differ")
    require(rerank.keys() == frozen.keys() == ab.keys(), "Raw frozen A/B IDs differ")
    require(audit.keys() == audited.keys() and answer.keys() == answered.keys(), "Raw review IDs differ")

    def same_fields(row, raw, keys):
        return all(row.get(key) == raw.get(key) for key in keys)

    for key, row in base.items():
        require(same_fields(row, generated[key],
                            ("count", "status", "pool_hit", "top5_hit", "used_hit", "fallback", "retrieved_ids"))
                and same_fields(row, graded.get(key, {}), ("accuracy", "grounded")), f"Thousand mismatch: {key}")
    for key, row in rerank.items():
        require(row["candidates"] == frozen[key]["candidates"]
                and same_fields(row, ab[key], ("count", "cohort", "pool_hit", "facet_only_hit"))
                and all(same_fields(row[arm], ab[key][arm], ("ids", "hit", "accuracy", "grounded"))
                        for arm in ("base", "new")), f"Frozen A/B mismatch: {key}")
    # The archived source contains the manual adjudications. Parse the literal without executing it.
    tree = ast.parse((raw_dir / "oa-audit-append.py").read_text(encoding="utf-8"))
    notes = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "notes" for target in node.targets))
    for key, row in audit.items():
        if key in notes:
            classification = notes[key][0]
        else:
            require(bool(audited[key]["facts"]) and all(fact["status"] == "selected"
                                                       for fact in audited[key]["facts"]), f"Unresolved audit: {key}")
            classification = "证据齐"
        require(row["classification"] == classification, f"Fact audit mismatch: {key}")
    for key, row in answer.items():
        require(same_fields(row, answered[key], ("cohort", "historical", "evidence_ids"))
                and all(same_fields(grade, answered[key]["grades"][arm], ("accuracy", "grounded"))
                        for arm, grade in row["grades"].items()), f"Answer A/B mismatch: {key}")


def replay(directory, raw_dir=None):
    base = read_latest(directory / "chinese-1000.jsonl")
    rerank = read_latest(directory / "rerank-ab-322.jsonl")
    audit = read_latest(directory / "fact-audit-43.jsonl")
    answer = read_latest(directory / "answer-ab-32.jsonl")
    require([len(item) for item in (base, rerank, audit, answer)] == [1000, 322, 43, 32],
            "Unexpected unique question counts")
    require(rerank.keys() <= base.keys(), "Frozen A/B contains an unknown question")
    cohorts = {name: [row for row in rerank.values() if row["cohort"] == name] for name in ("lost", "control")}
    lost = {key for key, row in base.items() if row["pool_hit"] == row["count"] > row["used_hit"]}
    require({row["id"] for row in cohorts["lost"]} == lost, "Frozen A/B lost cohort differs")
    require(len(cohorts["control"]) == 161
            and all(base[row["id"]]["used_hit"] == row["count"] for row in cohorts["control"]),
            "Frozen A/B control cohort differs")
    for group in cohorts.values():
        require(Counter(row["count"] for row in group) == {2: 58, 3: 103}, "Frozen A/B strata differ")
    for row in rerank.values():
        require(row["pool_hit"] == row["count"] == base[row["id"]]["count"]
                and len(row["candidates"]) == len(set(row["candidates"]))
                and all(set(row[arm]["ids"]) <= set(row["candidates"]) for arm in ("base", "new")),
                f"Frozen candidate mismatch: {row['id']}")
    rescued = {row["id"] for row in cohorts["lost"] if row["base"]["hit"] < row["count"] == row["new"]["hit"]}
    require(audit.keys() == rescued and answer.keys() == {key for key, row in audit.items()
                                                         if row["classification"] == "证据齐"}, "Audit cohort differs")
    for key, row in audit.items():
        revised = rerank[key]["new"]
        require(row["selected_ids"] == revised["ids"] and row["accuracy"] == revised["accuracy"],
                f"Audit selection differs: {key}")
    for key, row in answer.items():
        require(row["evidence_ids"] == audit[key]["selected_ids"] and row["historical"] == audit[key]["accuracy"],
                f"Answer A/B changed evidence: {key}")
    if raw_dir:
        verify_raw(raw_dir, base, rerank, audit, answer)
    return {
        "chinese_1000": {"overall": thousand(list(base.values())),
                         **{str(n): thousand([row for row in base.values() if row["count"] == n]) for n in (2, 3)}},
        "rerank_ab_322": {
            name: {"questions": len(group), "full_sources": transitions(group, lambda r, a: r[a]["hit"] == r["count"]),
                   "complete_correct": transitions(group, lambda r, a: r[a]["accuracy"] == "correct")}
            for name, group in cohorts.items()
        },
        "fact_audit_43": {
            name: {"questions": len(group), "complete_correct": sum(row["accuracy"] == "correct" for row in group)}
            for name in sorted({row["classification"] for row in audit.values()})
            for group in [[row for row in audit.values() if row["classification"] == name]]
        },
        "answer_ab_32": {
            name: {"questions": len(group), "baseline_correct": sum(row["grades"]["baseline"]["accuracy"] == "correct"
                                                                    for row in group),
                   "optimized_correct": sum(row["grades"]["optimized"]["accuracy"] == "correct" for row in group),
                   "grounded": {arm: sum(row["grades"][arm]["grounded"] is True for row in group)
                                for arm in ("baseline", "optimized")}}
            for name, group in (("overall", list(answer.values())),
                                ("control", [row for row in answer.values() if row["historical"] == "correct"]),
                                ("historical_miss", [row for row in answer.values() if row["historical"] != "correct"]))
        },
        "verified_raw_archive": raw_dir is not None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(Path(__file__).resolve().parent, args.raw_dir), ensure_ascii=False, indent=2))
