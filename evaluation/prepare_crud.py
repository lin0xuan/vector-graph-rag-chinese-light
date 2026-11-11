"""Recreate the fixed Chinese 1,000-question sample from a local CRUD-RAG checkout."""
import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET_REVISION = "1aace383994e1f68efa12cf2a8e2dadfb4102ceb"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_questions(split):
    documents = {}
    questions = []
    for count in (2, 3):
        eligible = []
        for index, row in enumerate(split[f"questanswer_{count}docs"]):
            texts = [row[f"news{j}"].strip() for j in range(1, count + 1)]
            question = row["questions"].strip()
            reference = row["answers"].strip()
            if 0 < len(question) <= 300 and reference and all(texts) and len(set(texts)) == count:
                eligible.append((index, row["ID"], question, reference, texts))
        for index, event_id, question, reference, texts in random.Random(20261110 + count).sample(eligible, 500):
            sources = [digest(text) for text in texts]
            documents.update(zip(sources, texts))
            questions.append({
                "id": f"{count}docs:{index}", "event_id": event_id,
                "question": question, "reference": reference, "sources": sources,
            })
    return documents, questions


def sample_distractors(paths, documents):
    sample = []
    rng = random.Random(20261111)
    seen = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                body = line.strip()
                if not body or digest(body) in documents:
                    continue
                seen += 1
                if len(sample) < 500:
                    sample.append(body)
                else:
                    position = rng.randrange(seen)
                    if position < 500:
                        sample[position] = body
    if len(sample) != 500:
        raise ValueError(f"Expected at least 500 eligible distractors, found {seen}")
    return {digest(body): body for body in sample}, seen


def write_jsonl(path, items):
    with path.open("x", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare the historical Chinese CRUD-RAG sample")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to CRUD-RAG's data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory for JSONL inputs")
    args = parser.parse_args()
    split_path = args.data_dir / "crud_split" / "split_merged.json"
    distractor_paths = sorted((args.data_dir / "80000_docs").glob("documents_dup_part_*"))
    documents, questions = sample_questions(json.loads(split_path.read_text(encoding="utf-8")))
    distractors, seen = sample_distractors(distractor_paths, documents)
    meta = {
        "expected_dataset_revision": DATASET_REVISION,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "questions": len(questions),
        "event_ids": len({test["event_id"] for test in questions}),
        "gold_documents": len(documents),
        "distractors": len(distractors),
        "eligible_distractor_rows": seen,
    }
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "documents.jsonl",
                ({"source": source, "title": "中文新闻", "text": body} for source, body in (documents | distractors).items()))
    write_jsonl(args.output_dir / "questions.jsonl", questions)
    (args.output_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
