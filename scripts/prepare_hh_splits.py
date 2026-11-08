"""Prepare reproducible HH-RLHF splits without changing the raw dataset."""
import csv
import hashlib
import json
import random
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

SEED = 42
VALIDATION_FRACTION = 0.10
MARKER = "\n\nAssistant:"


def prompt_id(prompt):
    text = unicodedata.normalize("NFC", prompt)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_pair(row):
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None, "missing_or_nonstring_text"
    cp, cm, cr = chosen.rpartition(MARKER)
    rp, rm, rr = rejected.rpartition(MARKER)
    if not cm or not rm:
        return None, "missing_assistant_marker"
    if cp != rp:
        return None, "different_last_turn_contexts"
    if not cr.strip() or not rr.strip():
        return None, "empty_final_response"
    if cr.strip() == rr.strip():
        return None, "identical_final_responses"
    # Retain response whitespace so prompt + response reconstructs the raw text.
    return {
        "prompt_id": prompt_id(cp + cm),
        "prompt": cp + cm,
        "chosen_response": cr,
        "rejected_response": rr,
    }, None


def make_splits(raw, seed=SEED, fraction=VALIDATION_FRACTION):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    reserved = set()
    # Reserve recognized contexts even from test rows excluded by our parser.
    for row in raw["test"]:
        for field in ("chosen", "rejected"):
            text = row.get(field)
            if isinstance(text, str):
                prefix, marker, _ = text.rpartition(MARKER)
                if marker:
                    reserved.add(prompt_id(prefix + marker))

    excluded = []
    clean = {}
    for original_split in ("train", "test"):
        kept, seen = [], set()
        for index, row in enumerate(raw[original_split]):
            example_id = f"{original_split}:{index}"
            record, reason = parse_pair(row)
            if reason is None and original_split == "train":
                if record["prompt_id"] in reserved:
                    reason = "prompt_reserved_for_official_test"
            if reason is None:
                pair_key = hashlib.sha256(json.dumps(
                    [row["chosen"], row["rejected"]], ensure_ascii=False
                ).encode("utf-8")).hexdigest()
                if pair_key in seen:
                    reason = "exact_duplicate_pair"
                else:
                    seen.add(pair_key)
            if reason:
                excluded.append({"example_id": example_id, "reason": reason})
            else:
                kept.append({"example_id": example_id, **record})
        clean[original_split] = kept

    groups = sorted({row["prompt_id"] for row in clean["train"]})
    if len(groups) < 2 or not clean["test"]:
        raise ValueError("Insufficient usable prompt groups or test examples.")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    validation_groups = set(groups[:count])
    splits = {"train": [], "validation": [], "test": clean["test"]}
    for row in clean["train"]:
        split = "validation" if row["prompt_id"] in validation_groups else "train"
        splits[split].append(row)

    group_sets = {key: {r["prompt_id"] for r in rows}
                  for key, rows in splits.items()}
    overlaps = {f"{a}_vs_{b}": len(group_sets[a] & group_sets[b])
                for a, b in (("train", "validation"), ("train", "test"),
                             ("validation", "test"))}
    if any(overlaps.values()) or (group_sets["train"] | group_sets["validation"]) & reserved:
        raise RuntimeError("Prompt overlap detected.")
    total_raw = sum(len(raw[s]) for s in ("train", "test"))
    if sum(map(len, splits.values())) + len(excluded) != total_raw:
        raise RuntimeError("Row accounting failed.")
    counts = {s: dict(Counter(e["reason"] for e in excluded
                             if e["example_id"].startswith(s + ":")))
              for s in ("train", "test")}
    report = {
        "seed": seed,
        "validation_fraction_of_remaining_prompt_groups": fraction,
        "input_rows": {s: len(raw[s]) for s in ("train", "test")},
        "output_rows": {s: len(rows) for s, rows in splits.items()},
        "unique_prompt_groups": {s: len(g) for s, g in group_sets.items()},
        "excluded_rows_by_reason": counts,
        "prompt_overlap": overlaps,
        "assignment_sha256": {
            s: hashlib.sha256("\n".join(sorted(r["example_id"] for r in rows))
                              .encode("utf-8")).hexdigest()
            for s, rows in splits.items()
        },
    }
    return splits, excluded, report


def main():
    from datasets import Dataset, DatasetDict, load_from_disk
    root = Path(__file__).resolve().parents[1]
    destination = root / "data" / "hh_rlhf_splits"
    if destination.exists():
        raise SystemExit("Splits already exist; keeping the existing files.")
    source = json.loads((root / "configs" / "dataset_source.json").read_text())
    raw = load_from_disk(str(root / "data" / "hh_rlhf_raw"))
    splits, excluded, report = make_splits(raw)
    report["dataset_revision"] = source["revision"]
    report["group_definition"] = "Full prompt including dialogue history; NFC and normalized line endings; outer whitespace stripped for grouping only."
    report["test_policy"] = "Retained rows come only from official test; format filtering and exact deduplication applied. All recognized official-test contexts reserved."

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="hh_split_") as temporary:
        stage = Path(temporary) / "dataset"
        DatasetDict({s: Dataset.from_list(rows) for s, rows in splits.items()}).save_to_disk(str(stage))
        with (stage / "excluded_rows.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["example_id", "reason"])
            writer.writeheader()
            writer.writerows(excluded)
        with (stage / "split_assignments.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["example_id", "split", "prompt_id"])
            for split, rows in splits.items():
                writer.writerows((r["example_id"], split, r["prompt_id"]) for r in rows)
        (stage / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(destination)
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nSPLIT SUMMARY")
    print(json.dumps(report, indent=2))
    print("\nSaved:", destination)


if __name__ == "__main__":
    main()
