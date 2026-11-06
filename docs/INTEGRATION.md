# Preservation-first migration into the user's EXISTING Git repository

Target: `~/projects/cKAN` at exact tested HEAD `5ef8e9f949a8f8ac91ca2167ccb4a7815a824abf`.
This package is **not** a replacement `.git` repository. `ckan_integration.patch` is computed
against the original uploaded `original_at_HEAD.zip` contents and preserves existing relative
source paths. The root file `src/main.cpp` remains Qt; the new CLI uses `src/cli_main.cpp`.

## Safer procedure: isolated worktree

1. Download the patch and migration helper into your Windows Downloads folder.
2. From WSL, locate both; commands below assume the two paths resolved explicitly.
3. Inspect `git status -sb`, ensure no local changes and the exact HEAD. If the branch has
   moved, stop; rebase/reconcile source changes rather than forcing this patch.
4. Run `bash ckan_integrate_worktree.sh /path/to/ckan_integration.patch`. The script creates
   `~/projects/cKAN-integration` as a **Git worktree** on `feat/ckan-unified-inference`.
   It uses `git apply --check` before changing files, never uses `--delete`, never commits
   automatically, never pushes, and never changes the original working tree.
5. In the resulting worktree run `cmake -S . -B build ...`, build, CTest, PyTorch parity and
   sanitizer checks. Review the original GUI wiring, legacy JSON, and CMake differences.
6. Optionally build Qt6 on WSL/desktop. The GUI has not been compiled in the packaging host.
7. Only after inspection: `git status -sb; git diff --check; git diff --stat; git diff`.
   Commit on the feature branch if correct. Keep `main` untouched until reviewed merge.

## Collision disposition

- `CMakeLists.txt`: replaced with a root CMake target graph that supports both model families
  and disables the old dummy CUDA kernel. Optional Qt target keeps `KANViewer`.
- `README.md`: replaces inaccurate CUDA/ONNX claims with tested feature inventory.
- `src/main.cpp`: **kept as Qt entrypoint**, repaired layout/GUI lives in `src/gui/`.
- `src/CMakeLists.txt`: historical stub; explicit targets prevent unsafe recursive globbing
  of `kan_cuda.cu` and extra main functions.
- `models/model_loader.*`: preserves the JSON schema, adds validation and an optional
  Boost.PropertyTree loader. The `KANModel` structs remain compatible with original JSON.
- `src/inference/kan_engine.*`: preserves the original mathematical architecture, repairs
  multilayer chaining and adds validation. This does not perform conversion to edge KAN.
- `src/inference/kan_cuda.*`: **unchanged original files** retained in Git, excluded from build.
- `tests/`, `tools/`, `include/`, `docs/`, `results/`: additive optimizations and evidence.

## Clean rollback

No repository-wide destructive commands are needed. Before the worktree is committed, you can
inspect, revert specific reviewed files inside the feature worktree, or remove the entire
worktree via `git worktree remove <path>` only after ensuring it contains no needed local
changes. Never delete the original `~/projects/cKAN` folder; never force push main.
