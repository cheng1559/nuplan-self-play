"""Export nuPlan log windows to normalized scene files, including reactive-agent inputs.

    uv run --env-file .env nuplan-export --case DATABASE TOKEN --output-dir data/scenes
    uv run --env-file .env nuplan-export --from-scenes data/scenes --overwrite

``--from-scenes`` re-exports every case named by the ``*.metadata.json`` files there.
``--cases FILE`` reads a JSON list of [database, token, scenario_type] (see
scripts/prepare_nuplan_splits.py); ``--shard K/N`` exports every N-th of them, so N
processes can split one list. Cases are ordered by log, so each process keeps reusing
nuPlan's cached map.
"""

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("NUPLAN_DATA_STORE", "local")

from cs5478_project.data.dataset import SCENE_FORMAT_VERSION, load_scene, save_scene  # noqa: E402
from cs5478_project.data.map_chunks import polyline_chunks  # noqa: E402
from cs5478_project.data.map_simplification import simplify_map  # noqa: E402
from cs5478_project.data.nuplan.scenario import (  # noqa: E402
    make_scenario,
    read_metadata,
    source_fields,
)
from cs5478_project.data.nuplan.source import (  # noqa: E402
    connect,
    load_frames,
    log_map_name,
    scenario_tags,
)
from cs5478_project.defaults import DEFAULT_DT, EPISODE_FRAMES, NPC_ROUTE_HORIZON_S  # noqa: E402


def export_case(
    database,
    token,
    output_dir,
    *,
    scenario_type=None,
    overwrite=False,
    min_future_s=45.0,
    compress=False,
):
    """Write ``nuplan-<token>.pt`` and its metadata sidecar; return a summary."""
    from cs5478_project.data.nuplan.map_geometry import build_map
    from cs5478_project.data.nuplan.rails import build_rails
    from cs5478_project.data.nuplan.routes import build_scene_routes
    from cs5478_project.data.nuplan.tracks import build_tracks

    database = Path(database)
    scenario_id = f"nuplan-{token}"
    destination = Path(output_dir) / f"{scenario_id}.pt"
    if destination.exists() and not overwrite:
        raise ValueError(f"Refusing to replace existing scene: {destination}")
    with connect(database) as conn:
        frames = load_frames(conn)
        index = next((i for i, f in enumerate(frames) if f.token_hex == token), None)
        if index is None:
            raise ValueError(f"Token {token} is not a lidar frame in {database}")
        anchor = frames[index]
        map_name = log_map_name(conn)
        tags = scenario_tags(conn, anchor.token)
        scenario_type = scenario_type or (tags[0] if tags else "unknown")
        scene, tokens, categories, track_info = build_tracks(
            conn, frames, index, scenario_id, min_future_s
        )
    scenario = make_scenario(
        database, token, map_name, scenario_type=scenario_type, timestamp=anchor.timestamp_us
    )
    origin = (anchor.x, anchor.y, anchor.yaw)
    map_data, map_info = build_map(scene, scenario.map_api, origin)
    # Chunks come from the unsimplified geometry; simplification only thins segments.
    chunk_points, chunk_type, chunk_pieces = polyline_chunks(map_data)
    map_info["chunks"] = len(chunk_type)
    map_data = replace(map_data, chunk_points=chunk_points, chunk_type=chunk_type)
    scene = replace(scene, map=map_data)
    routes, corrected = build_scene_routes(
        scene, scenario.map_api, scenario.get_route_roadblock_ids(), gt_routes=scene.routes
    )
    # Signals attach to final segments, so simplify before associating them.
    scene = replace(scene, routes=routes, map=simplify_map(scene.map))
    rails, segment_signal, chunk_signal, rail_info = build_rails(
        scene, scenario, tokens, categories, corrected, chunk_pieces
    )
    scene = replace(
        scene,
        rails=rails,
        map=replace(scene.map, segment_signal=segment_signal, chunk_signal=chunk_signal),
    )
    scene.validate()
    temporary = destination.with_suffix(".tmp.pt")
    save_scene(scene, temporary, compress=compress)
    verified = load_scene(temporary)
    if (
        verified.num_actors != scene.num_actors
        or verified.map.segments.shape != scene.map.segments.shape
    ):
        raise RuntimeError("Saved scene changed its actor slots or map on reload")
    metadata = {
        "data_origin": "nuplan",
        "scene_id": scenario_id,
        "scene_format_version": SCENE_FORMAT_VERSION,
        "source": {
            "database": str(database),
            "token": token,
            "scenario_type": scenario_type,
            "tags": tags,
            "map_name": map_name,
            "timestamp": anchor.timestamp_us,
        },
        "sampling": {
            "dt_s": DEFAULT_DT,
            "frames": EPISODE_FRAMES,
            "npc_route_max_s": NPC_ROUTE_HORIZON_S,
            "method": "Timestamp interpolation within contiguous annotations; nearest "
            "visibility within 51 ms; no interpolation across missing annotations",
        },
        "coordinates": {"world_from_scene": {"x": anchor.x, "y": anchor.y, "heading": anchor.yaw}},
        "actors": [
            {"slot": slot, "track_token": token_, "category": category}
            for slot, (token_, category) in enumerate(zip(tokens, categories, strict=True))
        ],
        "tracks": track_info,
        "map": map_info | {"segments": len(scene.map.segments), "simplification": "RDP 2048"},
        "routes": {
            "ego": "Hermes corrected map route",
            "npc": "Consecutive GT from first appearance, no extrapolation",
            "corrected_roadblock_ids": corrected,
        },
        "reactive": rail_info
        | {"source": "nuplan build_idm_agents_on_map_rails, idm_agents_observation defaults"},
    }
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(destination)
    return {"path": str(destination), "scenario_type": scenario_type, **rail_info}


def cases_from_scenes(directory):
    cases = []
    for path in sorted(Path(directory).glob("*.metadata.json")):
        fields = source_fields(
            read_metadata(path.with_name(path.name[: -len(".metadata.json")] + ".pt"))
        )
        cases.append((fields["database"], fields["token"], fields["scenario_type"]))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", nargs=2, action="append", default=[], metavar=("DB", "TOKEN"))
    parser.add_argument("--from-scenes", type=Path, help="Re-export the cases in this directory.")
    parser.add_argument("--cases", type=Path, help="JSON list of [database, token, type].")
    parser.add_argument("--shard", default="0/1", help="K/N: export cases K, K+N, ...")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --from-scenes or data/scenes.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Keep scenes already exported."
    )
    parser.add_argument(
        "--min-future",
        type=float,
        default=45.0,
        help="Seconds of log needed after the start; below 45 s, routes are shorter.",
    )
    parser.add_argument(
        "--compress", action="store_true", help="Deflate scene files (about 5x smaller)."
    )
    args = parser.parse_args(argv)
    cases = [(Path(db), token, None) for db, token in args.case]
    if args.from_scenes:
        cases += cases_from_scenes(args.from_scenes)
    if args.cases:
        listed = sorted(json.loads(args.cases.read_text()))
        shard, shards = (int(value) for value in args.shard.split("/"))
        if not 0 <= shard < shards:
            parser.error("--shard must be K/N with 0 <= K < N")
        # Contiguous blocks keep one log's cases (and its map) in one process.
        size = -(-len(listed) // shards)
        cases += [
            (Path(db), token, kind) for db, token, kind in listed[shard * size : (shard + 1) * size]
        ]
    if not cases:
        if args.cases:  # a shard past the end of the list has nothing to do
            return
        parser.error("Give --case DB TOKEN, --from-scenes DIR, or --cases FILE")
    output = args.output_dir or args.from_scenes or Path("data/scenes")
    failures = 0
    for database, token, scenario_type in cases:
        if args.skip_existing and (Path(output) / f"nuplan-{token}.pt").exists():
            continue
        try:
            summary = export_case(
                database,
                token,
                output,
                scenario_type=None if scenario_type in (None, "unknown") else scenario_type,
                overwrite=args.overwrite,
                min_future_s=args.min_future,
                compress=args.compress,
            )
            print(json.dumps(summary), flush=True)
        except Exception:  # report every case, then fail
            failures += 1
            print(f"FAILED {database} {token}", file=sys.stderr, flush=True)
            traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} of {len(cases)} cases failed")


if __name__ == "__main__":
    main()
