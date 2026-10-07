#!/usr/bin/env sh
# Prove it: run the whole pipeline in a Linux network namespace that has no
# network interface except loopback. If anything tried to phone home, it would fail.
# Needs `unshare` and permission to create namespaces (root, or a user namespace).
set -e
cd "$(dirname "$0")/.."
[ -f examples/soundscape.wav ] || sh scripts/get_example_audio.sh
OUT=$(mktemp -d)
unshare -n sh -c '
  printf "interfaces: "; tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d " " | tr "\n" " "; echo
  printf "internet reachable? "; (curl -s -m 4 -o /dev/null https://pypi.org && echo YES) || echo no
  export DAWN_CHORUS_DB="$1/life.db"
  python3 -I dawn_chorus.py plan   --lat 42.45 --lon -76.50 --out "$1/bingo.html"
  python3 -I dawn_chorus.py listen examples/soundscape.wav --lat 42.45 --lon -76.50 --out "$1/notes.html"
' sh "$OUT"
echo "outputs in $OUT"
