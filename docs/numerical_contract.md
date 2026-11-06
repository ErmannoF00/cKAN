# CKAN 1 numerical and file contract

## Model, units and parameterization

This repository executes *edge-wise, piecewise-linear spline KANs with an additional linear skip*. It is not a loader for arbitrary published KAN frameworks and it does not execute cubic B-splines. Each layer has `I` inputs, `O` outputs, `K >= 2` shared (per-layer) strictly increasing knots, output biases `b[o]`, skip coefficients `s[o,i]`, and edge coefficients `c[o,i,k]`.

For input `x[i]`, `u[i] = clamp(x[i], t[0], t[K-1])`. Let `k[i]=min(K-2, max(0, upper_bound(t,u[i])-1))` and `a[i]=(u[i]-t[k[i]])/(t[k[i]+1]-t[k[i]])`. The output is

```
y[o] = float32( double(b[o]) +
        sum_i ( double(s[o,i])*double(x[i]) +
                (1-a[i])*double(c[o,i,k[i]]) + a[i]*double(c[o,i,k[i]+1]) ) )
```

The actual parenthesization evaluates the two-term interpolation first, then adds it to the skip product and accumulates the *combined contribution* in `i=0,1,...,I-1` order. Reordering floating-point additions can alter results, and therefore none of the three engines uses `-ffast-math`. The same cast-to-double and cast-back-to-float32 policy applies to every layer. Each layer's output is the next layer's input, with no activation, softmax, ReLU or batch norm. Inputs and weights must be finite. Output overflow may produce `Inf` for pathological finite inputs; it is **not** silently clamped.

`std::upper_bound` is applied per edge only in the baseline. Both optimized engines decode a segment once per input and reuse that `(k,a)` across output channels. `packed` creates a transposed coefficient copy `[input][knot][output]` and skip copy `[input][output]`, allowing adjacent outputs to read contiguous coefficient values. These copies are created at executor construction and **consume additional memory**. The canonical model's data layout is always `[output][input][knot]`.

## File format: strict ASCII v1

```
CKAN 1
LAYERS L
LAYER I O K
KNOTS t0 ... tK-1
BIAS b0 ... bO-1
SKIP s00 ... s(O-1)(I-1)
COEFF c000 ... c(O-1)(I-1)(K-1)
[repeat LAYER through COEFF L times]
```

All numbers serialized using nine significant decimal digits (round-trip for finite float32), with `.` decimal point and classic stream locale. `I,O ∈ [1,4096]`, `K ∈ [2,256]`, `L ∈ [1,64]`; total coefficient count <= 4,000,000. The parser rejects NaN/Inf, bad lengths, disconnected layers, bad dimensions/knots, truncated data, and trailing tokens. It does **not** include a checksum, authentication, standardized metadata, dynamic knot training, arbitrary-B-spline degree, or framework checkpoint import. Treat `.kan` from an untrusted source as data, never as executable code; size limits are not a complete parser security audit.

## Testing tolerances

- Baseline ↔ shared ↔ packed C++: **bitwise identical float32** (the actual randomized-test requirement, with exact math order).
- PyTorch training reference ↔ C++: `max_abs_error <= 3e-5` on the 256 fixed-seed held-out samples. PyTorch uses float32 reduction and triangular basis; C++ uses double accumulator and direct two-knot interpolation. Tiny rounding differences are expected.
- Coverage: nonuniform knot grids, repeated execution with scratch buffers, multilayer chains, extrapolation/clamping, malformed files, invalid input, model serialization and deterministic example output. Random tests do not prove formal equivalence.

## Why shared decode does not reduce every memory access

For a layer with `I` inputs and `O` outputs, the baseline performs `I*O` interval searches while the two optimized engines perform `I`. All three still evaluate `I*O` edge splines, requiring two selected coefficient values and one skip coefficient per edge. Packed layout improves spatial locality for the output loop but does not reduce the mathematical number of spline coefficients. The current packing adds `4*I*O*(K+1)` bytes of stored copies, not an SRAM-CIM claim.
