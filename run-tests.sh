#!/usr/bin/env bash
# Run the ATLAS-Sandbox test suite in headless Chrome.
#
# There is no node on this machine, so the loop is: serve the repo over HTTP
# (ES modules need a real origin), load dev/tests.html in headless Chrome, and grep
# the DOM for the marker the harness writes. Exits nonzero if any test fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROME="${CHROME:-$(command -v google-chrome || command -v google-chrome-stable || command -v chromium)}"
PORT="${PORT:-8731}"
TIMEOUT="${TIMEOUT:-90}"

if [[ -z "$CHROME" ]]; then
  echo "run-tests: no Chrome binary found (set CHROME=/path/to/chrome)" >&2
  exit 127
fi

# Serve from a fixed port; if it is busy, walk upward until we find a free one.
for _ in $(seq 0 20); do
  if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
  PORT=$((PORT + 1))
done

python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT" >/dev/null 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

# Wait for the server to accept connections rather than sleeping a guess.
for _ in $(seq 0 100); do
  if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3<&-; break; fi
  sleep 0.05
done

PROFILE="$(mktemp -d)"
trap 'cleanup; rm -rf "$PROFILE"' EXIT

DOM="$(timeout "$TIMEOUT" "$CHROME" \
  --headless \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --no-first-run \
  --user-data-dir="$PROFILE" \
  --virtual-time-budget=60000 \
  --run-all-compositor-stages-before-draw \
  --dump-dom "http://127.0.0.1:$PORT/dev/tests.html?${1:-}" 2>/dev/null)"
CHROME_RC=$?

if [[ $CHROME_RC -ne 0 && $CHROME_RC -ne 1 ]]; then
  echo "run-tests: chrome exited $CHROME_RC (timeout or crash)" >&2
  [[ -n "$DOM" ]] && echo "$DOM" | head -40 >&2
  exit 1
fi

printf '%s' "$DOM" | python3 "$ROOT/tools/parse_dom.py"
RC=$?
if [[ $RC -eq 2 ]]; then
  printf '%s' "$DOM" | tail -20 >&2
fi
exit $RC
