#!/usr/bin/env bash
# adaptive-practice-loop — one entry point.
#
#   ./run.sh ui        start the demo at http://127.0.0.1:8000  (⚠️ this is the one
#                      to show a reviewer — it needs the API extra only)
#   ./run.sh serve     start the API alone
#   ./run.sh test      every self-test in the repo, then the test suite
#   ./run.sh all       test, then serve
#
# ⚠️ The core maths needs NOTHING beyond the standard library. `ui` and `serve` need
# fastapi + uvicorn; everything else runs on a fresh clone before anything is installed.
set -euo pipefail
cd "$(dirname "$0")"

# ⚠️⚠️ PICK THE INTERPRETER THAT CAN ACTUALLY RUN THIS, AND SAY WHICH ONE.
#
# The first version hardcoded `python3` while the author tested with a DIFFERENT
# venv, so every test passed and `./run.sh ui` failed on a clean machine with
# "fastapi/uvicorn are not installed". Testing a script with interpreter A and
# shipping a script that uses interpreter B is not a test.
#
# Order: explicit override, then this repo's venv, then anything on PATH.
PY=""
for cand in "${PY_OVERRIDE:-}" "./.venv/bin/python" "$(command -v python3 || true)"; do
  [ -n "$cand" ] && [ -x "$cand" ] && { PY="$cand"; break; }
done
[ -n "$PY" ] || { echo "  ⚠️  no python3 found on PATH" >&2; exit 1; }
echo "  python: $PY"

# ⚠️ The venv is preferred, but a venv without the dependency must not be chosen
# silently — fall back to any interpreter that HAS it before giving up.
has_api() { "$1" -c "import fastapi, uvicorn" 2>/dev/null; }
if ! has_api "$PY"; then
  for cand in "$(command -v python3 || true)" /usr/bin/python3; do
    if [ -n "$cand" ] && [ -x "$cand" ] && has_api "$cand"; then PY="$cand"; break; fi
  done
fi

need_api() {
  has_api "$PY" && return 0
  cat >&2 <<EOF

  ⚠️  The demo needs fastapi and uvicorn, and none was found for:
        $PY

  Set it up once:

      cd $(pwd)
      python3 -m venv .venv
      .venv/bin/pip install -r requirements.txt

  ⚠️ Everything else in this repo needs NOTHING beyond the standard library.
     './run.sh test' runs every self-test with no dependencies at all.
EOF
  exit 1
}

case "${1:-ui}" in
  ui|serve)
    need_api
    [ "$1" = "ui" ] && echo "  → http://127.0.0.1:8000   (Ctrl-C to stop)"
    exec "$PY" -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
    ;;
  test)
    for m in irt scheduler selector competency retrieval tutor idempotency langchain_pipeline vector_index; do
      [ -f "src/$m.py" ] && "$PY" "src/$m.py" --selftest || true
    done
    for t in tests/test_idempotency_http.py tests/test_workflow.py tests/test_pgvector.py; do
      [ -f "$t" ] && "$PY" "$t" || true
    done
    ;;
  all)
    "$0" test && exec "$0" serve
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
