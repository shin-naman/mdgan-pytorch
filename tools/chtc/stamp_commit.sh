#!/bin/bash
# Record the current commit in src/mdgan/_commit.json before copying the code to the cluster
# without .git, so provenance.json in every run still names the code that produced it.
# Run from the repository root:  bash tools/chtc/stamp_commit.sh
set -euo pipefail
commit=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then dirty=true; else dirty=false; fi
printf '{"git_commit": "%s", "git_dirty": %s}\n' "$commit" "$dirty" > src/mdgan/_commit.json
echo "stamped $(cat src/mdgan/_commit.json)"
if [ "$dirty" = true ]; then echo "warning: uncommitted changes; the stamp cannot reproduce them" >&2; fi