"""Closed-loop nuPlan devkit simulation of a SingleFrameNet checkpoint (NR-CLS, R-CLS).

Runs the devkit's own ``run_simulation`` with ``SingleFramePlanner`` and
``OneStageController`` (cs5478_project.nuplan_sim) on the scenarios of evaluation case
lists ([database, token, type] rows, as scripts/prepare_nuplan_splits.py writes), then
prints the devkit aggregator's final scores. The planner builds its inputs online from
the devkit, so nothing needs to be exported. Scenarios use the devkit's own extraction
(15 s starting 3 s before the token).

    uv run --env-file .env python scripts/nuplan_closed_loop.py CHECKPOINT \
        --eval val14=data/splits/val14.json --goal-horizon 2 --workers 16 \
        --output outputs/nuplan/run

Each set and setting is one devkit experiment under OUTPUT/<set>/<experiment>/; finished
ones are reused. ``summary.jsonl`` collects the scores.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = {"NR": "closed_loop_nonreactive_agents", "R": "closed_loop_reactive_agents"}


def write_configs(directory, name, cases, limit):
    """Scenario filter and scenario builder of a case list ([database, token, type] rows,
    as in data/splits); returns the number of scenarios.

    The builder lists the cases' own database files: an evaluation set can span several
    split directories, and a log can exist in more than one of them."""
    import nuplan.planning.script.config.common as common

    rows = sorted((token, Path(database), kind) for database, token, kind in cases)[:limit]
    if not rows:
        raise SystemExit(f"{name}: no cases")
    databases = sorted({str(database) for _, database, _ in rows})
    kinds = sorted({kind for _, _, kind in rows if kind not in (None, "unknown")})
    lines = [
        "_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_: 'all'",
        *(["scenario_types:", *(f"  - '{kind}'" for kind in kinds)] if kinds
          else ["scenario_types: null"]),
        "scenario_tokens:",
        *(f"  - '{token}'" for token, _, _ in rows),
        "log_names:",
        *(f"  - '{log}'" for log in sorted({database.stem for _, database, _ in rows})),
        "map_names: null",
        "num_scenarios_per_type: null",
        "limit_total_scenarios: null",
        "timestamp_threshold_s: null",
        "ego_displacement_minimum_m: null",
        "ego_start_speed_threshold: null",
        "ego_stop_speed_threshold: null",
        "speed_noise_tolerance: null",
        "expand_scenarios: false",
        "remove_invalid_goals: false",
        "shuffle: false",
    ]  # fmt: skip
    target = directory / "scenario_filter" / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    # The devkit's scenario_builder/nuplan.yaml with explicit database files.
    builder = (Path(common.__path__[0]) / "scenario_builder" / "nuplan.yaml").read_text()
    replaced = []
    for line in builder.splitlines():
        if line.startswith("data_root:"):
            line = f"data_root: '{Path(databases[0]).parent}'"
        elif line.startswith("db_files:"):
            line = "\n".join(["db_files:", *(f"  - '{database}'" for database in databases)])
        replaced.append(line)
    if not any(line.startswith("db_files:") for line in replaced):
        raise SystemExit("The devkit's scenario_builder/nuplan.yaml has no db_files entry")
    target = directory / "scenario_builder" / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(replaced) + "\n")
    return len(rows)


def final_scores(directory):
    """The aggregator's final row: score, scenario count, and per-metric means."""
    import pandas

    files = sorted((directory / "aggregator_metric").glob("*.parquet"))
    if not files:
        return None
    table = pandas.read_parquet(files[-1])
    final = table[table["scenario"] == "final_score"].iloc[0]
    row = {"score": float(final["score"]), "scenarios": int(final["num_scenarios"])}
    for name in (
        "no_ego_at_fault_collisions",
        "drivable_area_compliance",
        "ego_is_making_progress",
        "driving_direction_compliance",
        "time_to_collision_within_bound",
        "speed_limit_compliance",
        "ego_progress_along_expert_route",
        "ego_is_comfortable",
    ):
        if name in table.columns:
            row[name] = float(final[name])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        type=lambda value: tuple(value.split("=", 1)),
        metavar="NAME=CASES",
    )
    parser.add_argument("--settings", nargs="+", default=["NR", "R"], choices=("NR", "R"))
    parser.add_argument("--action", default="mode", choices=("mean", "mode"))
    parser.add_argument("--goal-horizon", type=float, help="Seconds ahead for the goal.")
    parser.add_argument(
        "--reward-weights",
        nargs="+",
        default=[],
        metavar="NAME=VALUE",
        help="Reward weights told to a reward-conditioned policy instead of the defaults.",
    )
    parser.add_argument("--limit", type=int, help="Only the first N cases of each set.")
    parser.add_argument("--workers", type=int, default=8, help="Processes; 1: sequential.")
    parser.add_argument("--ray", action="store_true", help="Ray workers instead of a process pool.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import nuplan.planning.script.run_simulation as run_simulation

    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    environment = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{os.environ.get('PYTHONPATH', '')}")
    environment.setdefault("NUPLAN_DATA_STORE", "local")
    searchpath = ",".join(
        (
            f"file://{REPO / 'configs' / 'nuplan_eval'}",
            f"file://{output / 'config'}",
            "pkg://nuplan.planning.script.config.common",
            "pkg://nuplan.planning.script.experiments",
        )
    )
    if args.workers <= 1:
        worker = ["worker=sequential"]
    elif args.ray:
        worker = ["worker=ray_distributed", f"worker.threads_per_node={args.workers}"]
    else:
        worker = [
            "worker=single_machine_thread_pool",
            "worker.use_process_pool=true",
            f"worker.max_workers={args.workers}",
        ]
    summary = []
    for name, cases in args.eval:
        filter_name = f"{name}_cases"
        count = write_configs(
            output / "config", filter_name, json.loads(Path(cases).read_text()), args.limit
        )
        for setting in args.settings:
            # The aggregator only reads metric files whose path names its challenge.
            directory = output / name / EXPERIMENTS[setting]
            scores = final_scores(directory)
            if scores is None:
                command = [
                    sys.executable,
                    run_simulation.__file__,
                    f"+simulation={EXPERIMENTS[setting]}",
                    "planner=single_frame_planner",
                    f"planner.single_frame_planner.checkpoint={checkpoint}",
                    f"planner.single_frame_planner.action={args.action}",
                    f"planner.single_frame_planner.device={args.device}",
                    "ego_controller=one_stage_controller",
                    f"scenario_builder={filter_name}",
                    f"scenario_filter={filter_name}",
                    # No metric_summary_callback: its PDF needs an old matplotlib.
                    "main_callback=[time_callback,metric_file_callback,metric_aggregator_callback]",
                    "callback=none",  # no per-scenario simulation logs (large)
                    *worker,
                    f"output_dir={directory}",
                    f"experiment_name={name}_{setting}",
                    f"hydra.searchpath=[{searchpath}]",
                ]
                if args.reward_weights:
                    weights = ",".join(w.replace("=", ":") for w in args.reward_weights)
                    command.append(f"planner.single_frame_planner.reward_weights={{{weights}}}")
                if args.goal_horizon is not None:
                    command.append(
                        f"planner.single_frame_planner.goal_horizon_s={args.goal_horizon}"
                    )
                print(f"== {name} {setting}: {count} scenarios -> {directory}", flush=True)
                with (output / f"{name}_{setting}.log").open("w") as log:
                    code = subprocess.run(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT
                    ).returncode
                scores = final_scores(directory)
                if scores is None:
                    print(f"   failed (exit {code}); see {output / f'{name}_{setting}.log'}")
                    continue
            row = {
                "checkpoint": str(checkpoint),
                "set": name,
                "setting": setting,
                "action": args.action,
                "listed": count,
                **scores,
            }
            summary.append(row)
            print(json.dumps(row), flush=True)
    (output / "summary.jsonl").write_text("".join(json.dumps(row) + "\n" for row in summary))
    print(f"{'set':12} {'setting':7} {'scenarios':>9} {'score':>7} collision progress")
    for row in summary:
        print(
            f"{row['set']:12} {row['setting']:7} {row['scenarios']:9} {row['score']:7.4f} "
            f"{row.get('no_ego_at_fault_collisions', float('nan')):9.3f} "
            f"{row.get('ego_progress_along_expert_route', float('nan')):8.3f}"
        )


if __name__ == "__main__":
    main()
