"""Print the checkpoint of a run directory with the best val14 score (mean of NR and R)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1])
scores = defaultdict(dict)  # update -> split -> (score, epoch)
for line in (run / "metrics.jsonl").open():
    row = json.loads(line)
    if row.get("kind") == "eval" and row["split"] in ("val14", "val14-reactive"):
        scores[row["update"]][row["split"]] = (row["score"], row["epoch"])
best = None
for update, splits in scores.items():
    epoch = splits["val14"][1] if "val14" in splits else None
    names = (f"update_{update:05d}.pt", f"epoch_{epoch:03d}.pt") if epoch is not None else ()
    paths = [run / "checkpoints" / name for name in names]
    path = next((path for path in paths if path.exists()), None)
    if path is None:
        continue
    value = sum(score for score, _ in splits.values()) / len(splits)
    if best is None or value > best[0]:
        best = (value, path, update, splits)
if best is None:
    sys.exit(f"{run}: no evaluated checkpoint")
detail = " ".join(f"{name} {score:.3f}" for name, (score, _) in sorted(best[3].items()))
print(f"{run.name}: update {best[2]} {detail}", file=sys.stderr)
print(best[1])
