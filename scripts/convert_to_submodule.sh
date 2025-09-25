#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/convert_to_submodule.sh <REMOTE_URL> [BRANCH]
# Example: scripts/convert_to_submodule.sh https://github.com/YOUR_GH_USER/zion-cryptonote.git zion-mainnet

REMOTE_URL="${1:-}"
BRANCH_NAME="${2:-zion-mainnet}"

if [[ -z "${REMOTE_URL}" ]]; then
  echo "Usage: $0 <REMOTE_URL> [BRANCH]" >&2
  exit 1
fi

if [[ ! -d "zion-cryptonote/.git" ]]; then
  echo "Error: zion-cryptonote is not an embedded git repo (missing .git)." >&2
  exit 1
fi

# Push the local branch to the provided remote
(
  cd zion-cryptonote
  git remote remove origin >/dev/null 2>&1 || true
  git remote add origin "${REMOTE_URL}"
  git push -u origin "${BRANCH_NAME}"
)

# Remove embedded repo from index but keep working copy
if git ls-files --error-unmatch zion-cryptonote >/dev/null 2>&1; then
  git rm --cached -r zion-cryptonote
fi

# Add as a proper submodule, pinned to the branch
# Note: Git 2.13+ supports -b to set branch in .gitmodules
if git submodule add -b "${BRANCH_NAME}" "${REMOTE_URL}" zion-cryptonote; then
  :
else
  # Fallback without -b, then patch .gitmodules
  git submodule add "${REMOTE_URL}" zion-cryptonote
  git config -f .gitmodules submodule.zion-cryptonote.branch "${BRANCH_NAME}"
fi

git add .gitmodules zion-cryptonote

git commit -m "chore: convert zion-cryptonote to proper submodule (branch ${BRANCH_NAME})."

echo "Done. To initialize on fresh clones, run:\n  git submodule update --init --recursive"