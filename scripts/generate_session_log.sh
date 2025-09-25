#!/usr/bin/env bash
set -euo pipefail

# generate_session_log.sh
# Usage:
#   scripts/generate_session_log.sh [--title "My note"] [--notes "extra notes"]
# Creates a dated session log under docs/sessions/YYYY-MM-DD/session-HHMMSS.md

TITLE=""
NOTES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)
      TITLE=${2:-}
      shift 2
      ;;
    --notes)
      NOTES=${2:-}
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

DATE_UTC=$(date -u +%Y-%m-%d)
TIME_UTC=$(date -u +%H%M%S)
DEST_DIR="docs/sessions/${DATE_UTC}"
DEST_FILE="${DEST_DIR}/session-${TIME_UTC}.md"

mkdir -p "$DEST_DIR"

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
LAST_COMMITS=$(git --no-pager log --oneline -n 15 2>/dev/null || true)
STATUS=$(git --no-pager status -sb 2>/dev/null || true)
REMOTE=$(git --no-pager remote -v 2>/dev/null | sed -n '1p' || true)
HOST_UNAME=$(uname -a 2>/dev/null || true)
PWD_NOW=$(pwd)
NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$DEST_FILE" <<EOF
# Session Log – ${DATE_UTC} ${TIME_UTC}Z

${TITLE:+> ${TITLE}}

Meta
- Time (UTC): ${NOW_ISO}
- Branch: ${BRANCH}
- Remote: ${REMOTE}
- Host: ${HOST_UNAME}
- PWD: ${PWD_NOW}

Git status
```
${STATUS}
```

Last commits
```
${LAST_COMMITS}
```

Notes
${NOTES}

EOF

echo "Created: ${DEST_FILE}"
