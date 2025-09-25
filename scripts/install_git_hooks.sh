#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

HOOKS_DIR=".git/hooks"
mkdir -p "$HOOKS_DIR"

cat > "$HOOKS_DIR/post-commit" <<'EOF'
#!/usr/bin/env bash
scripts/post_commit_cleanup.sh || true
EOF
chmod +x "$HOOKS_DIR/post-commit"

# post-push není standardní git hook, použijeme post-commit jako nejbližší.
echo "Git hooks installed: post-commit"
