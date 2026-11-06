# Research agenda: turning a CPU optimization into a meaningful hardware question

## Baseline novelty boundary

Shared knot-grid lookup and piecewise-linear interpolation are not new mathematical ideas. By 2026, published KAN accelerators already use FPGA LUT quantization and compute-in-memory strategies; KANELÉ (FPGA 2026), KA-CIM (2026), and spline-locality online learning (ICML 2026) are essential prior art. Do not claim first KAN accelerator, first LUT KAN, first CIM KAN or first shared-knot decoding.

## A defensible PhD experiment

**Hypothesis:** For memory-constrained KAN inference, an input-major layout and interval-once-per-input schedule may reduce knot-search overhead and improve coefficient-cache locality, but the benefit depends on `I,O,K`, sample batch, coefficient reuse and the model's memory hierarchy. Quantify when the packing overhead breaks even; compare to direct edge-major execution and a strong modern LUT/PWL implementation, not merely a deliberately naive Python loop.

1. Microarchitecture: define `K` knot entries in a small local ROM, `I` input feature decoders, and an `O_tile` vector of spline interpolators receiving a broadcast `(index, alpha)`. Store `[i,k,o]` contiguous in SRAM/BRAM banks; analyze port count, burst widths, bank conflicts, and scale at `I=8/32/128`, `O=8/32/128`, `K=8/16/32`.
2. Quantization: compare float32 baseline with a real fixed-point implementation, e.g. signed Q format for inputs/coefficients and suitable wider accumulators; specify rounding at each multiply, exact overflow behavior, and per-layer conversion. Re-run task accuracy and layer-wise numerical discrepancy. Never infer quantization benefits from a C++ type alias.
3. Hardware RTL: write a synthesizable input broadcast + segment decode + coefficient RAM + tile interpolator, with ready/valid, programmed weights, bounded loops and a scoreboarding testbench. Show **synthesis reports on a named FPGA**, including FF/LUT/BRAM/DSP, constraints, Fmax, cycle count, throughput. A C++ project alone does not prove a hardware architecture.
4. Co-design: evaluate model/coeff size, knot sharing options, output tile widths, on-chip bank capacity and coefficient packing. Compare exactly equivalent models and data, with separate model-load and inference latency.
5. Power: use implementation-based power analysis or device current measurements with documented toggles/activity. Exclude unsupported 'energy improvements' drawn from host runtime alone.

An actual paper-worthy mechanism would likely need stronger control than shared decode: e.g. autotuning between edge-major and input-major kernels based on `I,O,K` and cache size, variable-precision coefficient blocking with verified output bounds, or cross-layer scheduling that reduces buffer movement at equal model quality. Each alternative must be compared against modern literature rather than being asserted novel.

## Essential concrete research baselines

- Same trained KAN weights and inputs across baseline/shared/packed CPU paths.
- Uniform and nonuniform grids; check exact breakpoints and extrapolated inputs.
- Same numerical accuracy/parameter count versus MLP *and* a modern KAN implementation.
- Prepare once + process N samples: report break-even N; include packed-copy storage penalty.
- Batch `1, 16, 256`, workload widths `8,32,128`, knot counts `8,16,32`.
- A performance counter experiment (`perf stat`) for cache refs/misses and cycles where permitted.
- No comparisons of a float32 CPU baseline to an INT8 hardware kernel without accounting for accuracy, bitwidth, traffic and scheduling.
