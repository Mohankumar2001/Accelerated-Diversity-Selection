import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "Anthropic/hh-rlhf"
COMPONENTS = [
    "helpful-base",
    "helpful-rejection-sampled",
    "helpful-online",
    "harmless-base",
]
DATA_FILES = {
    split: [f"{part}/{split}.jsonl.gz" for part in COMPONENTS]
    for split in ("train", "test")
}

source_path = ROOT / "configs" / "dataset_source.json"
source_path.parent.mkdir(parents=True, exist_ok=True)

# Reuse the recorded revision on subsequent runs.
if source_path.exists():
    source = json.loads(source_path.read_text())
    if source["repo_id"] != REPO or source["data_files"] != DATA_FILES:
        raise ValueError("Existing dataset configuration differs from this script.")
else:
    revision = HfApi().dataset_info(REPO).sha
    if not revision:
        raise RuntimeError("Could not determine the dataset revision.")
    source = {
        "repo_id": REPO,
        "revision": revision,
        "data_files": DATA_FILES,
    }
    source_path.write_text(json.dumps(source, indent=2) + "\n")

print("Dataset revision:", source["revision"], flush=True)

dataset = load_dataset(
    REPO,
    revision=source["revision"],
    data_files=DATA_FILES,
    cache_dir=os.environ["HF_DATASETS_CACHE"],
)

def format_status(row):
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return "missing_or_nonstring_text"

    cp, cm, cr = chosen.rpartition("\n\nAssistant:")
    rp, rm, rr = rejected.rpartition("\n\nAssistant:")

    if not cm or not rm:
        return "missing_assistant_marker"
    if cp != rp:
        return "different_last_turn_contexts"
    if not cr.strip() or not rr.strip():
        return "empty_final_response"
    if cr.strip() == rr.strip():
        return "identical_final_responses"
    return "matching_context_nonempty_responses"

summary = {
    "repo_id": REPO,
    "revision": source["revision"],
    "splits": {},
}

for split, rows in dataset.items():
    summary["splits"][split] = {
        "rows": len(rows),
        "columns": rows.column_names,
        "format_checks": dict(Counter(format_status(row) for row in rows)),
    }

# Preserve all original rows; this stage only inspects them.
output_path = ROOT / "data" / "hh_rlhf_raw"
output_path.parent.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(output_path))

report_path = ROOT / "results" / "dataset_inspection.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(summary, indent=2) + "\n")

print("\nINSPECTION SUMMARY")
print(json.dumps(summary, indent=2))

# Preview training data only.
example = dataset["train"][0]
print("\nFIRST TRAINING EXAMPLE — abbreviated")
for field in ("chosen", "rejected"):
    print(f"\n{field}:")
    print(repr(example[field][:600]))

print("\nRaw dataset saved to:", output_path)
print("Inspection report saved to:", report_path)
