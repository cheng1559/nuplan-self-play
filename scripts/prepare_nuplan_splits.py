"""Write export case lists for training and evaluation (val14, test14-random, test14-hard).

Evaluation tokens come from the scenario-filter yamls in configs/nuplan_eval and are
located by scanning the mini, val, and test logs. Any training window that overlaps an
evaluation window in the same log is dropped, so evaluation scenes never appear in
training. Two training selections:

- ``spacing`` (train_mini.json): tagged frames at least ``--spacing`` seconds apart that
  leave room for the exporter's 45 s scenario.
- ``dense`` (train_mini_dense.json): the dense selection used for the report. Every log
  gets a 5 s time grid, plus
  tagged frames equisampled per scenario type with the per-type cap scaled to the number
  of logs (10,000 for all of trainval). Anchors need 4 s of log before and 20 s after,
  and skip each log's first two and last scenes. Export these with
  ``nuplan-export --min-future 20``.

    uv run --env-file .env python scripts/prepare_nuplan_splits.py --output data/splits
    uv run --env-file .env python scripts/prepare_nuplan_splits.py --selection dense
    uv run --env-file .env python scripts/prepare_nuplan_splits.py --selection dense \
        --train-splits train_boston train_pittsburgh --name train_partial

``--train-splits`` picks the training log directories (default: mini); the list is written
to ``<name>.json`` (default: ``train_mini`` or ``train_mini_dense``).
"""

import argparse
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

EVAL_SETS = {
    "val14": "configs/nuplan_eval/scenario_filter/val14.yaml",
    "test14": "configs/nuplan_eval/scenario_filter/test14_random.yaml",
    "test14_hard": "configs/nuplan_eval/scenario_filter/test14_hard.yaml",
}
SCENARIO_S = 45.2  # the exporter's NuPlanScenario length
EPISODE_S = 15.0
# Dense selection: about 383k type-capped + 540k grid anchors from 11,572 trainval logs.
DENSE_LOGS = 11572
DENSE_TYPE_CAP = 10000
DENSE_GRID_S = 5.0
DENSE_HISTORY_S = 4.0
DENSE_FUTURE_S = 20.0


def read_tokens(path):
    tokens, inside = [], False
    for line in Path(path).read_text().splitlines():
        if line.startswith("scenario_tokens:"):
            inside = True
        elif inside and line.strip().startswith("- "):
            tokens.append(line.strip()[2:].strip().strip("'\""))
        elif inside and line.strip() and not line.startswith(" "):
            inside = False
    return tokens


def connect(database):
    """A read-only connection that closes on exit (sqlite3's own context manager only
    commits, which leaks one descriptor per log when scanning thousands of them)."""
    return closing(sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True))


def find_tokens(database, wanted):
    """{token: (timestamp, scenario_type)} for wanted tokens found in this log."""
    with connect(database) as conn:
        rows = conn.execute(
            f"SELECT lp.token, lp.timestamp, st.type FROM lidar_pc AS lp "
            f"LEFT JOIN scenario_tag AS st ON st.lidar_pc_token = lp.token "
            f"WHERE lp.token IN ({','.join('?' * len(wanted))})",
            [bytes.fromhex(token) for token in wanted],
        ).fetchall()
    found = {}
    for token, timestamp, kind in rows:
        found.setdefault(bytes(token).hex(), (int(timestamp), kind or "unknown"))
    return found


def training_cases(database, spacing_s, blocked):
    """Tagged frames at least spacing_s apart with a full scenario ahead."""
    with connect(database) as conn:
        end = conn.execute("SELECT MAX(timestamp) FROM lidar_pc").fetchone()[0]
        rows = conn.execute(
            "SELECT lp.token, lp.timestamp, MIN(st.type) FROM scenario_tag AS st "
            "JOIN lidar_pc AS lp ON lp.token = st.lidar_pc_token "
            "GROUP BY lp.token ORDER BY lp.timestamp"
        ).fetchall()
    cases, last = [], None
    for token, timestamp, kind in rows:
        if (end - timestamp) / 1e6 < SCENARIO_S + 1.0:
            break
        if last is not None and (timestamp - last) / 1e6 < spacing_s:
            continue
        start, stop = timestamp, timestamp + EPISODE_S * 1e6
        if any(start < b and a < stop for a, b in blocked):
            continue
        cases.append([str(database), bytes(token).hex(), kind])
        last = timestamp
    return cases


def valid_scenes(conn):
    """Scene tokens except each log's first two and last (by name)."""
    names = conn.execute("SELECT token FROM scene ORDER BY name ASC").fetchall()
    return {bytes(token) for (token,) in names[2:-1]}


def dense_candidates(database, blocked):
    """(grid anchors, tagged anchors) of one log; tagged ones carry their scenario type."""
    with connect(database) as conn:
        rows = conn.execute(
            "SELECT token, timestamp, scene_token FROM lidar_pc ORDER BY timestamp"
        ).fetchall()
        tags = dict(
            conn.execute(
                "SELECT lidar_pc_token, MAX(type) FROM scenario_tag GROUP BY lidar_pc_token"
            ).fetchall()
        )
        scenes = valid_scenes(conn)
    if not rows:
        return [], []
    first, last = rows[0][1], rows[-1][1]

    def usable(token, timestamp, scene):
        if bytes(scene) not in scenes:
            return False
        if timestamp - first < DENSE_HISTORY_S * 1e6 or last - timestamp < DENSE_FUTURE_S * 1e6:
            return False
        stop = timestamp + EPISODE_S * 1e6
        return not any(timestamp < b and a < stop for a, b in blocked)

    grid, next_time = [], None
    for token, timestamp, scene in rows:
        if not usable(token, timestamp, scene):
            continue
        if next_time is None or timestamp >= next_time:
            grid.append([str(database), bytes(token).hex(), "grid"])
            next_time = timestamp + DENSE_GRID_S * 1e6
    tagged = [
        [str(database), bytes(token).hex(), str(tags[token])]
        for token, timestamp, scene in rows
        if token in tags and usable(token, timestamp, scene)
    ]
    return grid, tagged


def dense_cases(databases, blocked, workers, type_cap):
    """Grid anchors plus tagged anchors equisampled per type, deduplicated."""
    with ThreadPoolExecutor(workers) as pool:
        per_log = list(
            pool.map(lambda db: dense_candidates(db, blocked.get(str(db), [])), databases)
        )
    grid = [case for cases, _ in per_log for case in cases]
    by_type = {}
    for _, tagged in per_log:  # logs in order, frames by time
        for case in tagged:
            by_type.setdefault(case[2], []).append(case)
    seen = {case[1] for case in grid}
    typed = []
    for kind in sorted(by_type):
        cases = by_type[kind]
        if len(cases) > type_cap:
            cases = cases[:: max(len(cases) // type_cap, 1)][:type_cap]
        for case in cases:
            if case[1] not in seen:
                seen.add(case[1])
                typed.append(case)
    return grid, typed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--spacing", type=float, default=8.0, help="Seconds between starts.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--selection", choices=("spacing", "dense"), default="spacing")
    parser.add_argument(
        "--type-cap",
        type=int,
        help="dense per-type cap (default: 10,000 scaled to the number of logs).",
    )
    parser.add_argument(
        "--train-splits", nargs="+", default=["mini"], help="Training log directories."
    )
    parser.add_argument("--name", help="Output name for the training list.")
    args = parser.parse_args()
    data_root = os.path.expanduser(os.environ.get("NUPLAN_DATA_ROOT", "~/nuplan/dataset"))
    root = Path(data_root) / "nuplan-v1.1/splits"
    logs = {split: sorted((root / split).glob("*.db")) for split in ("mini", "val", "test")}
    train_logs = sorted(db for split in args.train_splits for db in (root / split).glob("*.db"))
    wanted = {name: read_tokens(path) for name, path in EVAL_SETS.items()}

    # Locate evaluation tokens, preferring mini logs, which the visualizer scenes use.
    eval_cases = {name: {} for name in wanted}
    with ThreadPoolExecutor(args.workers) as pool:
        for split in ("mini", "val", "test"):
            for name, tokens in wanted.items():
                missing = [t for t in tokens if t not in eval_cases[name]]
                if not missing:
                    continue
                results = pool.map(lambda db, m=missing: (db, find_tokens(db, m)), logs[split])
                for database, found in results:
                    for token, (timestamp, kind) in found.items():
                        eval_cases[name].setdefault(token, (str(database), timestamp, kind))
    args.output.mkdir(parents=True, exist_ok=True)
    blocked = {}
    for name, tokens in wanted.items():
        cases = [
            [eval_cases[name][t][0], t, eval_cases[name][t][2]]
            for t in tokens
            if t in eval_cases[name]
        ]
        for database, timestamp, _ in eval_cases[name].values():
            blocked.setdefault(database, []).append(
                (timestamp - EPISODE_S * 1e6, timestamp + SCENARIO_S * 1e6)
            )
        (args.output / f"{name}.json").write_text(json.dumps(cases, indent=0))
        print(f"{name}: {len(cases)} of {len(tokens)} tokens found", flush=True)

    if args.selection == "dense":
        name = args.name or "train_mini_dense"
        cap = args.type_cap or max(1, round(DENSE_TYPE_CAP * len(train_logs) / DENSE_LOGS))
        grid, typed = dense_cases(train_logs, blocked, args.workers, cap)
        train = grid + typed
        (args.output / f"{name}.json").write_text(json.dumps(train, indent=0))
        print(
            f"{name}: {len(train)} cases from {len(train_logs)} logs "
            f"({len(grid)} on the 5 s grid, {len(typed)} type-capped at {cap} per type)",
            flush=True,
        )
        return
    with ThreadPoolExecutor(args.workers) as pool:
        per_log = pool.map(
            lambda db: training_cases(db, args.spacing, blocked.get(str(db), [])), train_logs
        )
        train = [case for cases in per_log for case in cases]
    name = args.name or "train_mini"
    (args.output / f"{name}.json").write_text(json.dumps(train, indent=0))
    print(f"{name}: {len(train)} cases from {len(train_logs)} logs", flush=True)


if __name__ == "__main__":
    main()
