#!/usr/bin/env bash
set -euo pipefail

# save_ai_context.sh
# Uloží AI kontext/poznámky do datovaného logu v docs/sessions/YYYY-MM-DD/ai-session-HHMMSS.md
# Vstup může být přes --file <path>, --notes "text" nebo STDIN (pipe)
#
# Příklady:
#   echo "Dnešní konverzace a závěry..." | scripts/save_ai_context.sh --title "AI – shrnutí dne"
#   scripts/save_ai_context.sh --file /tmp/ai.txt --title "AI – build fixy"
#   scripts/save_ai_context.sh --notes "Krátká poznámka" --title "AI – quick note"

TITLE=""
FILE_INPUT=""
NOTES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)
      TITLE=${2:-}
      shift 2
      ;;
    --file)
      FILE_INPUT=${2:-}
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
DEST_FILE="${DEST_DIR}/ai-session-${TIME_UTC}.md"

mkdir -p "$DEST_DIR"

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
REMOTE=$(git --no-pager remote -v 2>/dev/null | sed -n '1p' || true)
HOST_UNAME=$(uname -a 2>/dev/null || true)
PWD_NOW=$(pwd)
NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

CONTENT=""

# 1) --file má přednost
if [[ -n "$FILE_INPUT" ]]; then
  if [[ -f "$FILE_INPUT" ]]; then
    CONTENT=$(cat "$FILE_INPUT")
  else
    echo "Soubor nenalezen: $FILE_INPUT" >&2
    exit 1
  fi
# 2) STDIN (pipe)
elif [ ! -t 0 ]; then
  CONTENT=$(cat)
# 3) fallback na --notes
else
  CONTENT="$NOTES"
fi

cat > "$DEST_FILE" <<EOF
# AI Session Log – ${DATE_UTC} ${TIME_UTC}Z

${TITLE:+> ${TITLE}}

Meta
- Time (UTC): ${NOW_ISO}
- Branch: ${BRANCH}
- Remote: ${REMOTE}
- Host: ${HOST_UNAME}
- PWD: ${PWD_NOW}

## Konverzace / Kontext
${CONTENT}

EOF

echo "Created: ${DEST_FILE}"