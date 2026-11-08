# HH-RLHF preparation

Run from the project root, in the `ee451` environment on a compute node:

```bash
source scripts/env.sh
python -m unittest discover -s tests -p 'test_prepare_hh_splits.py' -v
python -u scripts/prepare_hh_splits.py
```

The input is `data/hh_rlhf_raw`, created by `scripts/download_inspect_hh.py`.
Its source revision and input file names are recorded in `configs/dataset_source.json`.

## Fixed initial policy

- Require matching conversation contexts before the final `Assistant:` marker,
  and two nonempty final responses that differ after stripping outer whitespace.
  Exclude other pairs conservatively; the flag does not establish that their
  original annotations are wrong. Keep the original dataset unchanged.
- Remove exact duplicate ordered preference pairs within each original split.
- Reserve every recognized context in the original test split, including
  contexts from its excluded pairs. Remove matching training examples.
- Group the remaining training examples by full conversation context, including
  the final `Assistant:` marker. For grouping only, apply Unicode NFC,
  normalize line endings, and strip outer whitespace. Preserve internal spaces.
- With seed 42, choose 10% of those unique groups for validation (rounded,
  with both train and validation nonempty). This is not necessarily 10% of rows.
- Test examples originate only from the official test split. Report evaluation
  as using a filtered, deduplicated test split, not the entire official test set.

The validation fraction is separate from the project's later subset-selection
budget. All selection methods and the full-data reference must use the same
resulting training pool and held-out splits.

## Outputs

`data/hh_rlhf_splits` is a Hugging Face DatasetDict with train, validation, and
test splits. Each row includes `example_id`, `prompt_id`, `prompt`,
`chosen_response`, and `rejected_response`. Responses retain their original
whitespace: `prompt + chosen_response` reconstructs the preferred transcript.
An example ID such as `train:123` identifies row 123 of the original saved
training split. It depends on the pinned dataset and loading configuration.

The output directory also contains `excluded_rows.csv`, `split_assignments.csv`,
and `split_summary.json`. A small summary copy is saved under `results/` for Git.
Dataset text and large audit files remain under the Git-ignored `data` path.
Completed splits are not overwritten by rerunning the script.

All three pairwise prompt-overlap counts must be zero. This checks normalized
full-context overlap, not paraphrases or relationships between different
conversation contexts. Local tests cover parsing, reconstruction, duplicate
handling, group isolation, test reservation, determinism, and row accounting.
The real-data run and its output counts still need verification on CARC.
