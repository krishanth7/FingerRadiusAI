#!/usr/bin/env bash
# Run every language port against the shared fixture.
#
# The fixture holds inputs and the outputs the Python implementation produces.
# A port passes only if it reproduces them: gestures and finger flags exactly,
# radii and Kalman output to 1e-6. Missing toolchains are reported and skipped
# rather than failing, so this is useful on a machine with only some of them.
#
#   ./ports/run_conformance.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

FIXTURE="ports/fixtures/conformance.json"
[ -f "$FIXTURE" ] || { echo "Missing $FIXTURE - run tools/make_fixture.py"; exit 1; }

echo "Conformance: every port must reproduce the Python fixture"
echo "------------------------------------------------------------"

failed=0
ran=0

run() {
  local name="$1" probe="$2"; shift 2
  if ! command -v "$probe" >/dev/null 2>&1; then
    printf '  %-8s SKIP - %s is not installed\n' "$name" "$probe"
    return
  fi
  ran=$((ran + 1))
  if output=$("$@" 2>&1); then
    printf '  %-8s %s\n' "$name" "$(echo "$output" | tail -1)"
  else
    printf '  %-8s FAILED\n' "$name"
    echo "$output" | tail -12 | sed 's/^/      /'
    failed=$((failed + 1))
  fi
}

run python   python3  python3 ports/python_conformance.py
run ruby     ruby     ruby ports/ruby/conformance_test.rb
run r        Rscript  Rscript ports/r/conformance_test.R
# ports/go is its own Go module, so the test has to run from inside it.
run go       go       env -C ports/go go test ./... -count=1

echo "------------------------------------------------------------"
if [ "$failed" -eq 0 ]; then
  echo "All $ran available port(s) agree with Python."
  exit 0
fi
echo "$failed of $ran port(s) disagree with Python."
exit 1
