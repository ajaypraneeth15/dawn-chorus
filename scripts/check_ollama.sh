#!/usr/bin/env bash
# Preflight check for Dawn Chorus's OPTIONAL Gemma journal (`listen --journal`).
#
#   1. Is the Ollama server running on this machine?
#   2. Is a Gemma model installed locally?
#
# Exit codes:  0 = ready   1 = Ollama not reachable   2 = no Gemma model installed   3 = curl missing
#
# Override the address with:  DAWN_CHORUS_OLLAMA_URL=http://127.0.0.1:11434 bash scripts/check_ollama.sh
# (Windows: run this in Git Bash or WSL. Without bash, `ollama list` in PowerShell shows the same information.)
set -u

URL="${DAWN_CHORUS_OLLAMA_URL:-http://localhost:11434}"
URL="${URL%/}"

if [ -t 1 ]; then GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'; else GREEN=""; RED=""; YELLOW=""; RESET=""; fi
pass() { printf '%s[PASS]%s %s\n' "$GREEN" "$RESET" "$1"; }
fail() { printf '%s[FAIL]%s %s\n' "$RED" "$RESET" "$1"; }
warn() { printf '%s[WARN]%s %s\n' "$YELLOW" "$RESET" "$1"; }

if ! command -v curl >/dev/null 2>&1; then
  fail "curl is not installed, so this script cannot talk to Ollama."
  echo "       Install curl, or run 'ollama list' by hand to see your models."
  exit 3
fi

# --noproxy '*': a server on this same computer must never be reached through a proxy.
http_get() { curl --noproxy '*' -s -m 5 "$@"; }

# 1) Is the server up? A running Ollama answers GET / with HTTP 200.
code="$(http_get -o /dev/null -w '%{http_code}' "$URL/" 2>/dev/null || true)"
if [ "$code" != "200" ]; then
  fail "Ollama is not reachable at $URL (HTTP status: ${code:-none})."
  echo
  echo "  What to do:"
  echo "    * Open the Ollama app (Windows/macOS), or"
  echo "    * run this in another terminal and leave it open:   ollama serve"
  echo "    * then run this check again."
  echo "  Not installed yet? Get it from https://ollama.com"
  exit 1
fi
pass "Ollama server is running at $URL"

# 2) Which models are installed? /api/tags lists them as {"models":[{"name":"gemma3:1b",...}]}.
tags="$(http_get "$URL/api/tags" 2>/dev/null || true)"
models="$(printf '%s' "$tags" | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*:[[:space:]]*"\(.*\)"/\1/')"

if [ -z "$models" ]; then
  fail "Ollama is running, but no models are installed yet."
  echo
  echo "  What to do (this downloads about 1 GB, so use a network you are happy to spend data on):"
  echo "    ollama pull gemma3:1b"
  echo "  Other Gemma sizes and versions: https://ollama.com/library"
  exit 2
fi

gemma="$(printf '%s\n' "$models" | grep -i 'gemma' || true)"
if [ -z "$gemma" ]; then
  fail "No Gemma model is installed. Installed models:"
  printf '%s\n' "$models" | sed 's/^/         - /'
  echo
  echo "  What to do:   ollama pull gemma3:1b      (or any Gemma tag from https://ollama.com/library)"
  echo "  Dawn Chorus would otherwise fall back to your first model: $(printf '%s\n' "$models" | head -n1)"
  exit 2
fi

pass "Gemma model(s) installed locally:"
printf '%s\n' "$gemma" | sed 's/^/         - /'

# Models whose names end in -cloud run on Ollama's servers, not on this computer.
if printf '%s\n' "$gemma" | grep -qi -- '-cloud'; then
  warn "A model name ending in '-cloud' usually runs on Ollama's servers, not locally."
  echo "       For a fully local run, choose a Gemma model without '-cloud' in its name."
fi

first="$(printf '%s\n' "$gemma" | head -n1)"
echo
pass "Ready. Try it:"
echo "         python3 dawn_chorus.py listen walk.m4a --lat 42.45 --lon -76.50 --journal --ollama-model $first"
exit 0
