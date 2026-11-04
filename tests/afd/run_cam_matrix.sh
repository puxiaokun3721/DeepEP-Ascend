#!/usr/bin/env bash
# CAM AFD single-round matrix driver (docs/cam-deepep-test-plan.md §3.2).
# Run inside the A5 runtime container after the CAM run package and whl are installed.
# Every case is one `torchrun` invocation; multi-round and performance stay out of scope.
#
#   OUT=./cam_matrix_logs ./tests/afd/run_cam_matrix.sh            # all C01-C13
#   ONLY=C01,C06 DRY=1 ./tests/afd/run_cam_matrix.sh               # show commands only
#   WEIGHTS=signed ONLY=C01 ./tests/afd/run_cam_matrix.sh          # override weight group
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
TORCHRUN=${TORCHRUN:-torchrun}
WEIGHTS=${WEIGHTS:-unit}
OUT=${OUT:-./cam_matrix_logs}
ONLY=${ONLY:-}
DRY=${DRY:-0}
TIMEOUT_S=${TIMEOUT_S:-1800}

# id world tokens hidden topk experts routes extra
CASES=(
  "C01 8 4096 7168 6 256 balanced"
  "C02 8 1024 7168 6 256 balanced"
  "C03 8 2048 7168 6 256 balanced"
  "C04 8 8192 7168 6 256 balanced"
  "C05 8 16384 7168 6 256 balanced"
  "C06 2 4096 7168 6 256 balanced"
  "C07 4 4096 7168 6 256 balanced"
  "C08 8 4096 7168 1 256 balanced"
  "C09 8 4096 7168 2 256 balanced"
  "C10 8 4096 7168 4 256 balanced"
  "C11 8 4096 7168 8 256 balanced"
  "C12 8 1024 7168 6 256 balanced --all-masked"
  "C13 4 4096 7168 6 256 hotspot"
)

wanted() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}

mkdir -p "$OUT" 2>/dev/null || true
fail=0
ran=0
for spec in "${CASES[@]}"; do
  read -r id world tokens hidden topk experts routes extra <<EOF
$spec
EOF
  wanted "$id" || continue
  extra=${extra:-}
  for field in world tokens hidden topk experts; do
    eval "val=\$$field"
    case "$val" in ''|*[!0-9]*) echo "bad case spec: $spec" >&2; exit 2 ;; esac
  done
  cmd=("$TORCHRUN" --standalone "--nproc_per_node=$world" "$HERE/run_pipeline_cam.py"
       --batch "$tokens" --window "$tokens" --hidden "$hidden" --topk "$topk"
       --experts "$experts" --routes "$routes" --weights "$WEIGHTS"
       --timeout-seconds "$((TIMEOUT_S - 120))" --run-tag "$id")
  [ -n "$extra" ] && cmd+=($extra)
  if [ "$DRY" = "1" ]; then
    printf '%s\n' "${cmd[*]}"
    continue
  fi
  echo "=== $id world=$world tokens=$tokens topk=$topk experts=$experts routes=$routes weights=$WEIGHTS"
  timeout "$TIMEOUT_S" "${cmd[@]}" >"$OUT/$id.log" 2>&1
  rc=$?
  ran=$((ran + 1))
  if [ $rc -ne 0 ]; then
    echo "$id FAIL rc=$rc log=$OUT/$id.log"
    fail=$((fail + 1))
  elif grep -q '"status": "FUNCTION_PASS"' "$OUT/$id.log" || grep -q '"status":"FUNCTION_PASS"' "$OUT/$id.log"; then
    echo "$id FUNCTION_PASS"
  else
    echo "$id NO_FUNCTION_PASS log=$OUT/$id.log"
    fail=$((fail + 1))
  fi
done

if [ "$DRY" = "1" ]; then
  exit 0
fi
echo "matrix done: ran=$ran fail=$fail"
[ "$fail" -eq 0 ]
