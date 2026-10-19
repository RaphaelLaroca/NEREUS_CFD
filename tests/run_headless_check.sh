#!/usr/bin/env bash
# Headless mode: a run driven by the config file alone, with no window.
#
# Unlike the suites this compiles nothing: headless IS the application, so the check drives the program run_all.sh
# built into tests/work/bin. Usage: run_headless_check.sh [binary]. Every run happens in tests/work/headless.
set -u
TESTS="$(cd "$(dirname "$0")" && pwd)"

BIN="${1:-$TESTS/work/bin/nereus_headless}"
[ -x "$BIN" ] || { echo "FAIL: no $BIN -- run tests/run_all.sh, which builds it."; exit 1; }
echo "binary: $BIN"

WORK="$TESTS/work/headless"
rm -rf "$WORK"; mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0
check() {  # check <description> <expected> <actual>
  if [ "$2" = "$3" ]; then PASS=$((PASS+1)); printf '  ok    %s\n' "$1"
  else FAIL=$((FAIL+1)); printf '  FAIL  %s (expected %s, got %s)\n' "$1" "$2" "$3"; fi
}
# A copy of <in> that stops on its own after <n> steps: STOP_TIME = n dt (dt = 0.005, fixed in the fixture), no final
# export (it would add a frame). There is no --steps flag: the end of a run is in its file.
until_step() {  # until_step <in> <out> <n>
  sed -e 's/^STOP_CONDITION = .*/STOP_CONDITION = TIME/' \
      -e "s/^STOP_TIME = .*/STOP_TIME = $(awk "BEGIN{print $3*0.005}")/" \
      -e 's/^STOP_EXPORT = .*/STOP_EXPORT = 0/' "$1" > "$2"
}
contains() {  # contains <description> <file> <text>
  if grep -qF -- "$3" "$2"; then PASS=$((PASS+1)); printf '  ok    %s\n' "$1"
  else FAIL=$((FAIL+1)); printf '  FAIL  %s (no "%s" in output)\n' "$1" "$3"; fi
}

# Every run happens in the work directory, so the exports land there. The preset carries its scene and its PRESSURE block, so nothing else
# from the working directory is needed.
cp "$TESTS/configs/headless_base.cfg" "$WORK/base.cfg"   # a fixture: a guard must not read a user preset
cd "$WORK"

# ---------------------------------------------------------------- 1. usage
echo "1. usage and the refusal to run forever"
"$BIN" --help > help.log 2>&1; check "--help exits 0" 0 $?
contains "--help lists the options" help.log "--progress N"

"$BIN" --headless base.cfg > never.log 2>&1; check "STOP_CONDITION = NEVER is refused" 1 $?
contains "  and says why" never.log "Nothing would stop this run"

"$BIN" --headless base.cfg --wrong > bad.log 2>&1; check "an unknown option is refused" 1 $?

# ---------------------------------------------------------------- 2. a step count, from the file
echo "2. a run ends where its file says"
until_step base.cfg steps.cfg 40
"$BIN" --headless steps.cfg --progress 20 > steps.log 2>&1
check "exits 0" 0 $?
contains "runs exactly the steps the final time gives" steps.log "steps    40"
contains "writes a progress line" steps.log "step       20"
"$BIN" --headless steps.cfg --steps 40 > oldflag.log 2>&1; check "--steps is no longer accepted" 1 $?

# ---------------------------------------------------------------- 3. the config's own stopping rule
echo "3. the stopping rule out of the config"
sed -e 's/^STOP_CONDITION = .*/STOP_CONDITION = TIME/' \
    -e 's/^STOP_TIME = .*/STOP_TIME = 0.1/' base.cfg > stop.cfg
"$BIN" --headless stop.cfg --progress 0 > stop.log 2>&1
check "exits 0" 0 $?
contains "stops at the final time" stop.log "reached t = 0.100000"
contains "  and says the rule fired" stop.log "the stopping rule fired"
check "the final field was exported" 1 \
  "$(ls Exports/stop/64_2D_re100/VTK/*.vtr 2>/dev/null | wc -l | tr -d ' ')"
check "telemetry was written" 1 \
  "$([ -f Exports/stop/64_2D_re100/telemetry.csv ] && echo 1 || echo 0)"

# ---------------------------------------------------------------- 4. the export keys
echo "4. what to export, from the config"
until_step base.cfg base10.cfg 10
sed 's/^EXPORT_ENABLED = .*/EXPORT_ENABLED = 0/' base10.cfg > noexport.cfg
"$BIN" --headless noexport.cfg --progress 0 > noexport.log 2>&1
check "EXPORT_ENABLED = 0 writes nothing" 0 \
  "$([ -d Exports/noexport ] && echo 1 || echo 0)"

sed -e 's/^EXPORT_GRID = .*/EXPORT_GRID = 1/' \
    -e 's/^EXPORT_GRID_INTERVAL = .*/EXPORT_GRID_INTERVAL = 5/' base.cfg > frames0.cfg
until_step frames0.cfg frames.cfg 20
"$BIN" --headless frames.cfg --progress 0 > frames.log 2>&1
check "EXPORT_GRID writes one frame per interval" 4 \
  "$(ls Exports/frames/64_2D_re100/VTK/*.vtr 2>/dev/null | wc -l | tr -d ' ')"

# The same run twice has to produce the same fields: nothing in the loop may depend on wall-clock
# time or on how often a progress line was printed.
mkdir -p first && cp Exports/frames/64_2D_re100/VTK/*.vtr first/
rm -rf Exports/frames
"$BIN" --headless frames.cfg --progress 3 > frames2.log 2>&1
if diff -rq first Exports/frames/64_2D_re100/VTK > /dev/null 2>&1; then
  PASS=$((PASS+1)); echo "  ok    two runs of the same config agree byte for byte"
else
  FAIL=$((FAIL+1)); echo "  FAIL  two runs of the same config differ"
fi

# ---------------------------------------------------------------- 5. checkpoints
echo "5. checkpoints"
sed -e 's/^CHECKPOINT_AUTOSAVE = .*/CHECKPOINT_AUTOSAVE = 1/' \
    -e 's/^CHECKPOINT_INTERVAL = .*/CHECKPOINT_INTERVAL = 10/' \
    -e 's|^CHECKPOINT_PATH = .*|CHECKPOINT_PATH = auto.ckpt|' base.cfg > ckpt0.cfg
until_step ckpt0.cfg ckpt.cfg 25
"$BIN" --headless ckpt.cfg --progress 0 > ckpt.log 2>&1
check "auto-save writes the file" 1 "$([ -f auto.ckpt ] && echo 1 || echo 0)"

# ---------------------------------------------------------------- 6. interruption
# SIGINT finishes the step in progress, writes the checkpoint and leaves through the normal
# shutdown -- which on the GPU paths is also what runs the AMGX finalize.
echo "6. SIGINT finishes the step and checkpoints"
sed -e 's|^CHECKPOINT_PATH = .*|CHECKPOINT_PATH = interrupted.ckpt|' base.cfg > long0.cfg
until_step long0.cfg long.cfg 1000000
"$BIN" --headless long.cfg --progress 0 > sig.log 2>&1 &
RUN=$!
sleep 3
kill -INT $RUN
wait $RUN; SIGCODE=$?
check "exits 3" 3 $SIGCODE
contains "says it was interrupted" sig.log "interrupted"
check "wrote a checkpoint" 1 "$([ -f interrupted.ckpt ] && echo 1 || echo 0)"

# ---------------------------------------------------------------- 7. a field that goes non-finite
# NaN is the case that matters and the one a peak-velocity test misses: a running std::max LOSES
# every comparison against NaN, so a field full of them reports a tidy finite peak. MAC::AllFinite
# looks at the values.
echo "7. a non-finite field is caught"
until_step base.cfg base100.cfg 100
sed 's|^SCENE_END|initial u="sqrt(0-1)" v="0"\nSCENE_END|' base100.cfg > nan.cfg
"$BIN" --headless nan.cfg --progress 10 > nan.log 2>&1
check "exits 2" 2 $?
contains "says what happened" nan.log "went non-finite"

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]
