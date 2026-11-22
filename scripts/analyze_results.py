"""Benchmark tables and paired bootstrap comparisons from devkit closed-loop results.

Each ``--run NAME=DIR`` points to the ``--output`` directory of
``scripts/nuplan_closed_loop.py`` for one checkpoint (``DIR/summary.jsonl`` and
``DIR/<set>/<experiment>/aggregator_metric/*.parquet``). Writes ``results.csv`` (scores
and sub-metrics per run, set and setting), ``paired_bootstrap.csv`` (differences on the
same scenarios with 95% bootstrap intervals) and ``by_scenario_type.csv``.

    uv run python scripts/analyze_results.py --out outputs/analysis \
        --run replay=outputs/nuplan/replay --run selfplay=outputs/nuplan/selfplay \
        --run selfplay_dr=outputs/nuplan/selfplay_dr --pair selfplay:replay
"""

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = {"NR": "closed_loop_nonreactive_agents", "R": "closed_loop_reactive_agents"}
METRICS = [
    "score", "no_ego_at_fault_collisions", "drivable_area_compliance",
    "driving_direction_compliance", "ego_is_making_progress", "time_to_collision_within_bound",
    "ego_progress_along_expert_route", "speed_limit_compliance", "ego_is_comfortable",
]  # fmt: skip
SAMPLES = 20000


def per_scenario(directory: Path, name: str, setting: str) -> pd.DataFrame:
    pattern = directory / name / EXPERIMENTS[setting] / "aggregator_metric" / "*.parquet"
    files = glob.glob(str(pattern))
    if len(files) != 1:
        raise SystemExit(f"expected one aggregator file at {pattern}, found {len(files)}")
    frame = pd.read_parquet(files[0])
    return frame[frame["num_scenarios"].isna()].set_index("scenario")


def bootstrap(diff: np.ndarray, rng) -> tuple[float, float]:
    means = diff[rng.integers(0, len(diff), (SAMPLES, len(diff)))].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument(
        "--pair", action="append", default=[], metavar="A:B",
        help="Report A minus B (default: every pair of runs, later minus earlier).",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, scenes = [], {}
    for run, directory in runs.items():
        directory = Path(directory)
        for line in (directory / "summary.jsonl").open():
            record = json.loads(line)
            frame = per_scenario(directory, record["set"], record["setting"])
            scenes[run, record["set"], record["setting"]] = frame
            rows.append({
                "run": run, "set": record["set"], "setting": record["setting"],
                "scenarios": len(frame),
                "se": frame["score"].std(ddof=1) / np.sqrt(len(frame)),
                **{key: record[key] for key in METRICS},
            })  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "results.csv", index=False)
    wide = table.pivot(index="run", columns=["set", "setting"], values="score") * 100
    print(wide.loc[list(runs)].round(2).to_string())

    pairs = [tuple(item.split(":", 1)) for item in args.pair] or [
        (b, a) for a, b in itertools.combinations(runs, 2)
    ]
    rng = np.random.default_rng(5478)
    comparisons = []
    for a, b in pairs:
        for (run, name, setting), left in scenes.items():
            if run != a or (b, name, setting) not in scenes:
                continue
            right = scenes[b, name, setting]
            common = left.index.intersection(right.index)
            diff = (left.loc[common, "score"] - right.loc[common, "score"]).to_numpy()
            low, high = bootstrap(diff, rng)
            comparisons.append({
                "a": a, "b": b, "set": name, "setting": setting, "n": len(diff),
                "diff": diff.mean(), "lo": low, "hi": high,
            })  # fmt: skip
        for setting in EXPERIMENTS:
            parts = [
                (scenes[a, s, m]["score"] - scenes[b, s, m]["score"]).dropna().to_numpy()
                for (run, s, m) in scenes
                if run == a and m == setting and (b, s, m) in scenes
            ]
            if parts:
                diff = np.concatenate(parts)
                low, high = bootstrap(diff, rng)
                comparisons.append({
                    "a": a, "b": b, "set": "pooled", "setting": setting, "n": len(diff),
                    "diff": diff.mean(), "lo": low, "hi": high,
                })  # fmt: skip
    comparison = pd.DataFrame(comparisons)
    comparison["significant"] = (comparison.lo > 0) | (comparison.hi < 0)
    comparison.to_csv(args.out / "paired_bootstrap.csv", index=False)
    print("\npaired differences a - b (x100), 95% bootstrap interval")
    for r in comparison.itertuples():
        mark = "*" if r.significant else " "
        print(f"  {r.a} - {r.b} {r.set:12} {r.setting:3} {r.diff * 100:+6.2f} "
              f"[{r.lo * 100:+6.2f}, {r.hi * 100:+6.2f}] {mark} n={r.n}")  # fmt: skip

    # Scenario types on Val14, reactive traffic, for the first pair.
    a, b = pairs[0]
    if (a, "val14", "R") in scenes and (b, "val14", "R") in scenes:
        x, y = scenes[a, "val14", "R"], scenes[b, "val14", "R"]
        types = pd.DataFrame({
            "type": x["scenario_type"], a: x["score"], b: y["score"].reindex(x.index),
        }).groupby("type").agg(n=(a, "size"), **{a: (a, "mean"), b: (b, "mean")})  # fmt: skip
        types["diff"] = types[a] - types[b]
        types.sort_values("diff").to_csv(args.out / "by_scenario_type.csv")


if __name__ == "__main__":
    main()
