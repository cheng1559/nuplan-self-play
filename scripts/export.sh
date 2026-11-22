#!/usr/bin/env bash
# export.sh CASES OUTPUT PARALLEL [nuplan-export options...]
# Export a case list with PARALLEL processes (each needs ~1.1 GB of memory and one core),
# then retry once, then count. Logs go to OUTPUT.logs/. Safe to rerun: it skips scenes
# already exported. Run from the repository root.
#   scripts/export.sh data/splits/val14.json data/nuplan/val14 16
#   scripts/export.sh data/splits/train_full.json data/nuplan/train_full 96 \
#       --min-future 20 --compress
set -u
cases=$1; out=$2; parallel=$3; shift 3
set -a; . ./.env; set +a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
logs="$out.logs"; mkdir -p "$out" "$logs"
shards=$((parallel * 8))
for pass in 1 2; do
    echo "pass $pass: $shards shards, $parallel at a time ($(date))"
    seq 0 $((shards - 1)) | xargs -P "$parallel" -I{} sh -c \
        "uv run nuplan-export --cases $cases --shard {}/$shards --output-dir $out \
         --skip-existing $* > $logs/pass$pass-{}.log 2>&1 || true"
done
total=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$cases" 2>/dev/null \
        || uv run python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$cases")
done_count=$(ls "$out" | grep '\.pt$' | grep -vc '\.tmp\.pt$')
echo "exported $done_count of $total cases into $out ($(date))"
echo "failure reasons (data problems such as 'route has no usable map baselines' are expected, ~2-3%):"
grep -h "Error" "$logs"/pass2-*.log | sed 's/.*Error: //' | sort | uniq -c | sort -rn | head
