#!/usr/bin/env bash

# Lightweight monitor for Zion RPC shim via proxy or direct endpoint.
# - Polls height and uptime from /shim/metrics.json (or JSON-RPC getheight fallback)
# - Detects stalls (no height increase for N minutes) and high error rates
# - Optional webhook notifications via WEBHOOK_URL (POST JSON)
#
# Usage:
#   monitor-shim.sh [--url <base_url>] [--interval <sec>] [--stall-min <min>] [--log <file>]
# Defaults:
#   URL: http://localhost:8080 (proxy base)  => metrics at /shim/metrics.json
#   Interval: 30s, Stall: 5min
#
# Examples:
#   ./scripts/monitor-shim.sh --url http://91.98.122.165:8080
#   URL=http://91.98.122.165:18089 ./scripts/monitor-shim.sh  # direct shim (no /shim prefix)

set -euo pipefail

URL_DEFAULT=${URL:-"http://localhost:8080"}
INTERVAL=30
STALL_MIN=5
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL_DEFAULT="$2"; shift 2 ;;
    --interval)
      INTERVAL="$2"; shift 2 ;;
    --stall-min)
      STALL_MIN="$2"; shift 2 ;;
    --log)
      LOG_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

BASE_URL="$URL_DEFAULT"

has_jq() { command -v jq >/dev/null 2>&1; }

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() {
  local line="[$(ts)] $*"
  echo "$line"
  if [[ -n "$LOG_FILE" ]]; then echo "$line" >> "$LOG_FILE"; fi
}

notify() {
  if [[ -n "${WEBHOOK_URL:-}" ]]; then
    curl -sS -X POST -H 'Content-Type: application/json' \
      -d "{\"msg\":$(printf '%s' "$*" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"ts\": \"$(ts)\"}" \
      "$WEBHOOK_URL" >/dev/null || true
  fi
}

# Returns: height or empty on failure
get_height() {
  local url="$1"
  local metrics_url rpc_url height raw

  # Try metrics first
  if [[ "$url" =~ /shim/?$ || "$url" =~ :8080$ || "$url" =~ /$ ]]; then
    metrics_url="${url%/}/shim/metrics.json"
  else
    metrics_url="${url%/}/metrics.json"
  fi

  raw=$(curl -fsS --max-time 5 "$metrics_url" 2>/dev/null || true)
  if [[ -n "$raw" ]]; then
    if has_jq; then
      height=$(printf '%s' "$raw" | jq -r '(.height // .result.height // .metrics.height // empty) | numbers' 2>/dev/null || true)
    else
      height=$(printf '%s' "$raw" | grep -E '"height"\s*:\s*[0-9]+' -m1 | sed -E 's/.*"height"\s*:\s*([0-9]+).*/\1/' || true)
    fi
    if [[ "$height" =~ ^[0-9]+$ ]]; then echo "$height"; return 0; fi
  fi

  # Fallback: JSON-RPC getheight
  if [[ "$url" =~ :8080$ ]]; then
    rpc_url="${url%/}/shim/json_rpc"
  elif [[ "$url" =~ /shim/?$ ]]; then
    rpc_url="${url%/}/json_rpc"
  else
    rpc_url="${url%/}/json_rpc"
  fi

  raw=$(curl -fsS --max-time 5 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"getheight"}' "$rpc_url" 2>/dev/null || true)
  if [[ -n "$raw" ]]; then
    if has_jq; then
      height=$(printf '%s' "$raw" | jq -r '.result.height // empty' 2>/dev/null || true)
    else
      height=$(printf '%s' "$raw" | grep -E '"height"\s*:\s*[0-9]+' -m1 | sed -E 's/.*"height"\s*:\s*([0-9]+).*/\1/' || true)
    fi
    if [[ "$height" =~ ^[0-9]+$ ]]; then echo "$height"; return 0; fi
  fi

  echo "" # failure
}

last_height=""
last_change_epoch=$(date +%s)
stall_seconds=$((STALL_MIN*60))

log "Monitoring $BASE_URL (interval=${INTERVAL}s, stall=${STALL_MIN}min)"

while true; do
  start=$(date +%s)
  height=$(get_height "$BASE_URL")
  if [[ -z "$height" ]]; then
    log "WARN no height (endpoint unreachable or invalid JSON)"
    notify "WARN: no height from $BASE_URL"
  else
    if [[ -z "$last_height" || "$height" -gt "$last_height" ]]; then
      if [[ -n "$last_height" && "$height" -gt "$last_height" ]]; then
        dt=$((start - last_change_epoch))
        log "Height advanced: $last_height -> $height (+$((height-last_height))) after ${dt}s"
      else
        log "Height: $height"
      fi
      last_height="$height"
      last_change_epoch="$start"
    else
      age=$((start - last_change_epoch))
      if (( age > stall_seconds )); then
        log "ALERT height stalled at $height for ${age}s (> ${stall_seconds}s)"
        notify "ALERT: height stalled at $height for ${age}s on $BASE_URL"
      else
        log "No change (height=$height, ${age}s since last)"
      fi
    fi
  fi

  end=$(date +%s)
  elapsed=$((end - start))
  sleep_for=$((INTERVAL - elapsed))
  (( sleep_for < 1 )) && sleep_for=1
  sleep "$sleep_for"
done
