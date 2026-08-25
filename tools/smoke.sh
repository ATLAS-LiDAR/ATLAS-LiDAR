#!/usr/bin/env bash
# Smoke-test the page in headless Chrome and FAIL on any console error or
# exception, instead of relying on eyeballing a screenshot.
#
# This exists because a ReferenceError in the render path produced two blank
# canvases that a screenshot showed as "empty scene" rather than "broken" - the
# unit tests were all green because the bug was in UI wiring, which they do not
# cover. Any change to src/ui should be run through this.
#
#   tools/smoke.sh "" "scene=nuscenes-scene0" "mode=remove&forceActive=1"
PORT=${PORT:-8801}
(python3 -m http.server $PORT --bind 127.0.0.1 >/dev/null 2>&1 &)
sleep 1.5
rc=0
for Q in "$@"; do
  PROF=$(mktemp -d)
  OUT=$(timeout 150 google-chrome --headless --no-sandbox --disable-gpu \
        --disable-dev-shm-usage --user-data-dir=$PROF --virtual-time-budget=20000 \
        --enable-logging=stderr --log-level=0 \
        --dump-dom "http://127.0.0.1:$PORT/index.html?$Q" 2>&1)
  ERRS=$(printf '%s' "$OUT" | grep -iE 'ERROR:CONSOLE|Uncaught|ReferenceError|TypeError' | grep -v 'favicon' | head -3)
  ST=$(printf '%s' "$OUT" | python3 -c "
import sys,re,html
d=sys.stdin.read()
m=re.search(r'<div id=\"status\" class=\"([^\"]*)\">(.*?)</div>', d, re.S)
print((m.group(1) if m else '?')+' | '+' '.join(html.unescape(re.sub(r'<[^>]+>',' ',m.group(2))).split())[:120] if m else 'NO STATUS')
")
  if [ -n "$ERRS" ]; then
    echo "FAIL [$Q]"; echo "$ERRS" | sed 's/^/    /'; rc=1
  else
    echo "ok   [$Q] $ST"
  fi
  rm -rf "$PROF"
done
pkill -f "http.server $PORT"
exit $rc
