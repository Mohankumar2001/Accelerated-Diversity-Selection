#!/usr/bin/env bash
set -euo pipefail

# Run on an allocated GPU compute node after loading gcc and cuda modules.
# The default target is the V100 (compute capability 7.0). For another GPU,
# pass its capability without a dot, e.g. 80 for A100 or 86 for A40.
EE451_ARCH="${1:-70}"
if [[ ! "$EE451_ARCH" =~ ^[0-9]{2,3}$ ]]; then
    echo "Usage: bash scripts/run_compile_checks.sh [architecture, e.g. 70]" >&2
    exit 2
fi
EE451_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EE451_ROOT"
mkdir -p build/environment_checks results/environment
EE451_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EE451_LOG="results/environment/compile_checks_${EE451_STAMP}_${SLURM_JOB_ID:-local}.log"

# pipefail preserves compiler/test failures while tee saves the session log.
{
    printf 'UTC timestamp: %s\n' "$EE451_STAMP"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID:-unset}"
    printf 'Host: '
    hostname
    printf 'CUDA target: sm_%s\n' "$EE451_ARCH"
    printf 'Visible GPU devices: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    if type module >/dev/null 2>&1; then module list; fi
    command -v g++
    g++ --version
    command -v nvcc
    nvcc --version
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

    g++ -O2 -std=c++17 -Wall -Wextra -pedantic tests/cpu_smoke.cpp \
        -o build/environment_checks/cpu_smoke
    ./build/environment_checks/cpu_smoke

    nvcc -O2 -std=c++17 -ccbin "$(command -v g++)" -arch="sm_${EE451_ARCH}" \
        tests/cuda_smoke.cu -o build/environment_checks/cuda_smoke
    ./build/environment_checks/cuda_smoke
    printf 'ALL COMPILATION CHECKS PASSED\n'
} 2>&1 | tee "$EE451_LOG"

printf 'Saved log: %s\n' "$EE451_LOG"
