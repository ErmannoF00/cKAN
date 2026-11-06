# Benchmark protocol: reproducible, not marketing numbers

## Commands

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/kan_bench --reps 5 --rounds 9 --batch 32 --output results/bench_local.csv
python3 tools/plot_benchmark.py --input results/bench_local.csv --output results/bench_local.png
```

`kan_bench` constructs identically seeded random one-layer models for `(I,O)=(8,8),(32,32),(128,128)` and `K=8,16,32`, using strictly increasing **nonuniform** knot coordinates. Inputs are generated once from the same seed; all engines process identical vectors, parameters and float32 outputs. Each measurement creates a new executor (prep timer), a reusable workspace, warms up two batches, then collects nine independent blocks. Each block contains `reps * batch` serial, single-threaded inferences. `median_ns_per_sample` and `p95_block_ns_per_sample` are **block-average** latency statistics, not the p95 distribution of individual requests. No model loading/CSV parsing is timed in the steady-state inference blocks.

Preparation cost is separately timed from executor construction; this includes copying the canonical model for all engines and additionally transposing packed arrays in `packed`. It is not disk parse time. Output is accumulated in a volatile-observable checksum printed to CSV; the work may not be optimized away. All engines share identical workspace reuse. No CPU affinity or governor pinning is performed automatically: measurements can shift with clock frequency, process placement, background tasks and cache contention. A single host's results do not establish general hardware speedups.

For a fixed vector, interval decoding is O(I*O*logK) in the baseline and O(I*logK) with sharing, while every engine still needs O(I*O) edge evaluation; a packed executor adds O(I*O*K) coefficient copying at setup. With sufficiently many repeated inferences the setup can amortize, but **the break-even point depends on model and machine**. Packed copies increase host memory consumption; a better hardware representation could store only the packed arrangement and eliminate the duplicate, but that is *not* what this executable currently does.

Record processor name, OS, compiler version, build flags, turbo/governor, all raw CSV rows, model shapes and seeds. Use `perf stat` for cache misses/instructions if the host permits it; avoid claiming cache-miss reductions without actual counters. No GPU, FPGA, CIM, chip power or energy results are provided here.

## Quality comparisons

`tools/train.py` uses the same 1024 training and 256 test samples across models. The KAN has 173 *trainable* parameters and 26 additional fixed knot coordinates (13 for each layer), while the tanh MLP `2→43→1` has 173 trainable parameters. They receive equal Adam step counts and the same learning rate, but the choice of hyperparameters does not establish a globally fair architecture comparison. The affine control is fit by least squares. Report all model results, including when the MLP wins; no general model superiority can be inferred from this simple synthetic regression.
