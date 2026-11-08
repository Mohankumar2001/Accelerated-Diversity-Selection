# C++ and CUDA environment checks

Run inside an allocated GPU session from the project root:

```bash
module load gcc/13.3.0 cuda/12.6.3
bash scripts/run_compile_checks.sh 70
```

The target `70` is the V100's compute capability (7.0). Rebuild with the
appropriate target if the allocated GPU changes. The runner records the
node, job, compiler versions, CUDA module information when available, GPU
information, and test results under `results/environment/`. Executables go
under the existing Git-ignored `build/` directory.

The C++ check compiles and executes a serial integer sum. The custom CUDA
check allocates device buffers, copies inputs, launches a vector-add kernel,
synchronizes, copies results back, and checks every element against the CPU
calculation. Its input size has a partial final thread block. All runtime
API calls and the kernel launch are checked for errors.

These are environment smoke tests, not performance benchmarks or k-center
correctness tests. The CUDA kernel still needs to be compiled and executed
on CARC; this bundle's preparation environment has no CUDA compiler/GPU.
