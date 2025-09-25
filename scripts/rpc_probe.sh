#!/usr/bin/env bash
set -euo pipefail
ADDR=${1:-"ajmqontZjiVUmtNjQu1RNUYq1RZgd5EDodX3qgjcaTMoMzG8EkG4bVPgLhEgudBoH82fQU1iZVw6XPfddKWAHDdA3x92ToH4uo"}
HOST=${2:-"91.98.122.165"}
PORT=${3:-"18081"}

curl -s http://$HOST:$PORT/getinfo | jq . || true

for rs in 4 8 16 32 64 128 255 256 300 400; do
  echo "\n--- getblocktemplate reserve_size=$rs ---"
  curl -s -X POST http://$HOST:$PORT/json_rpc \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":"0","method":"getblocktemplate","params":{"wallet_address":"'"$ADDR"'","reserve_size":'"$rs"'}}' | jq . || true
  sleep 1
done
