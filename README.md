# CUDA-Accelerated Diversity Selection for Preference Data

EE 451 final project — USC.

Team: Katherine, Surpreet, and Mohan.

## Objective

Implement CPU and custom CUDA greedy k-center selection for
HH-RLHF preference data. Compare selector performance and evaluate
reward-model quality using selected subsets, random subsets,
and a full-data reference.

## Structure

- src/cpu: CPU selector
- src/cuda: CUDA selector
- src/data: preprocessing and embeddings
- src/models: reward-model training and evaluation
- scripts: execution and Slurm scripts
- configs: experiment configurations
- tests: correctness checks
- docs: setup instructions and project notes
- results: small benchmark tables and summaries

Datasets, embeddings, caches, and model checkpoints are excluded
from Git. Environment setup and dataset preparation are in progress.
