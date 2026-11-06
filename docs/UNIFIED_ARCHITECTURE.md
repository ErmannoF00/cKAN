# The preserved cKAN architecture and the new execution path

The original GUI (`src/main.cpp`, `src/gui/MainWindow.*`) now hosts two **separate**
mathematical models. The original CUDA squaring prototype is excluded from the build.

```text
                                    Qt6 KANViewer (original source paths)
                                             |
                       +---------------------+---------------------+
                       |                                           |
               Open legacy JSON                                 Open CKAN 1
              models/dummy_model.json                      models/trained_regression.kan
                       |                                           |
             validateKANModel()                          kan::Model::validate()
                       |                                           |
                 KANEngine                                kan::Executor
                       |                              +------------+------------+
           for each layer l:                     per-edge      shared        packed
             dense W1 * x                        interval      interval      interval
                 |                               per edge      per input     per input
            phi_h(z_h)                               |             |             |
                 |                              original       original       in,k,out
             dense W2 * h                         layout        layout        memory
                       |                              +------------+------------+
                       |                                           |
                  float32 vector                              float32 vector
                       +---------------------+---------------------+
                                             |
                                      output table
```

## Invariants

- A `.json` file is **not** accepted as `.kan` and vice versa. No implicit conversion.
- `KANEngine` retains `float` arithmetic and clamps the original hidden spline inputs.
- `kan::Executor` uses double edge accumulation and clamps spline inputs only, while
  linear skip inputs remain unclamped.
- Both sources can be trained/loaded separately; never compare their weights as if equal.
- Original `kan_cuda.cu` performs `out[i]=in[i]^2`. It is not used for inference.
- For performance claims, compare baseline/shared/packed on **identical** CKAN1 weights.
- For accuracy claims, compare against independent PyTorch plus a matched-size MLP.
