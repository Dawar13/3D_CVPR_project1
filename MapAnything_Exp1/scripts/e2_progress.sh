#!/usr/bin/env bash
# Progress of the E2-C eval run (run ON THE VM): one bar per job, from the per-run files written so far.
cd ~/3D-CVPR_Exp/MapAnything_Exp1
bar() {  # name done total
  local w=30 f=$(( $2 * 30 / ($3 > 0 ? $3 : 1) ))
  printf "%-26s %s%s %5d / %-5d %3d%%\n" "$1" "$(printf '%*s' "$f" '' | tr ' ' '#')" \
    "$(printf '%*s' $(( w - f )) '' | tr ' ' '.')" "$2" "$3" $(( 100 * $2 / ($3 > 0 ? $3 : 1) ))
}
n() { ls "$1/extras" 2>/dev/null | wc -l; }
a=$(n results/e2c_eval); b=$(n results/e2c_n4_eval); c=$(n results/e2c_n20_eval)
bar "C main grid (24 clips)" "$a" 3168
bar "C5 4-view clips (10)" "$b" 40
bar "C5 20-view clips (6)" "$c" 24
bar "E2-C eval total" $(( a + b + c )) 3232
f=$(cat results/e2c_*_eval/failures.jsonl results/e2c_eval/failures.jsonl 2>/dev/null | wc -l)
echo "failures: $f    GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
if [ -f results/e2c_eval/per_run.csv ] && [ "$a" -gt 20 ]; then
  first=$(stat -c %Y "$(ls -tr results/e2c_eval/extras/* | head -1)"); now=$(date +%s)
  echo "rate: $(( (now - first) / a )) s per inference -> about $(( (3232 - a - b - c) * (now - first) / a / 60 )) min left"
fi
