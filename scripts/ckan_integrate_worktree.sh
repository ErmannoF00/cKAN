#!/usr/bin/env bash
# Apply a reviewed, exact-base cKAN patch in a new Git worktree.
# Never overwrites or resets ~/projects/cKAN and never pushes/commits.
set -euo pipefail
EXPECTED='5ef8e9f949a8f8ac91ca2167ccb4a7815a824abf'
BASE="${CKAN_REPO:-$HOME/projects/cKAN}"
DEST="${CKAN_WORKTREE_PATH:-$HOME/projects/cKAN-integration}"
BRANCH='feat/ckan-unified-inference'
PATCH="${1:-}"
if [[ -z "$PATCH" || ! -f "$PATCH" ]]; then
  printf 'Usage: bash ckan_integrate_worktree.sh /absolute/path/ckan_integration.patch\n' >&2
  exit 2
fi
PATCH="$(realpath "$PATCH")"
if [[ ! -d "$BASE/.git" && ! -f "$BASE/.git" ]]; then
  echo "Not a Git checkout: $BASE" >&2; exit 2
fi
if [[ "$(git -C "$BASE" rev-parse --show-toplevel)" != "$(realpath "$BASE")" ]]; then
  echo 'Repository path is not its Git root.' >&2; exit 2
fi
HEAD="$(git -C "$BASE" rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: HEAD $HEAD is different from expected $EXPECTED" >&2
  echo 'Do not force-apply to an unknown snapshot.' >&2
  exit 2
fi
if [[ -n "$(git -C "$BASE" status --porcelain --untracked-files=all)" ]]; then
  echo 'STOP: original checkout is not clean; review/backup local changes.' >&2
  exit 2
fi
if [[ -e "$DEST" ]]; then
  echo "STOP: worktree destination already exists: $DEST" >&2; exit 2
fi
if git -C "$BASE" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "STOP: branch already exists: $BRANCH" >&2; exit 2
fi
mkdir -p "$(dirname "$DEST")"
git -C "$BASE" worktree add -b "$BRANCH" "$DEST" "$HEAD"
if ! git -C "$DEST" apply --check --index "$PATCH"; then
  echo 'Patch does not match exact-base files. Worktree is kept for inspection.' >&2
  exit 2
fi
git -C "$DEST" apply --index --binary "$PATCH"
git -C "$DEST" diff --cached --check
printf '\nPatch applied and STAGED in isolated Git worktree:\n  %s\n' "$DEST"
printf 'Original working-tree contents were not modified. No commit or push performed.\n'
printf '\nNext commands:\n'
printf '  cd "%s"\n' "$DEST"
printf '  git diff --cached --stat\n'
printf '  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n'
printf '  cmake --build build -j2\n'
printf '  ctest --test-dir build --output-on-failure\n'
printf '  # Review staged changes and commit ONLY after passing every gate.\n'
