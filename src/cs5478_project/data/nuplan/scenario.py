"""Rebuild the nuPlan scenario behind an exported scene from its metadata sidecar."""

import json
import os
import sqlite3
from pathlib import Path

DEFAULT_DATA_ROOT = "~/nuplan/dataset"  # nuPlan's default; set NUPLAN_DATA_ROOT in .env


def metadata_path(scene_path: str | Path) -> Path:
    return Path(scene_path).with_suffix(".metadata.json")


def read_metadata(scene_path: str | Path) -> dict:
    path = metadata_path(scene_path)
    if not path.is_file():
        raise ValueError(f"Scene metadata does not exist: {path}")
    return json.loads(path.read_text())


def source_fields(metadata: dict) -> dict:
    """Normalize the two exporter metadata layouts to database, token, map, and type."""
    source = metadata["source"]
    database = Path(source["database"])
    token = source.get("token") or source.get("scenario_token")
    map_name = source.get("map_name") or source.get("map")
    if not token or not map_name:
        raise ValueError("Scene metadata lacks a scenario token or map name")
    return {
        "database": database,
        "token": token,
        "map_name": map_name,
        "scenario_type": source.get("scenario_type", "unknown"),
        "timestamp": source.get("timestamp"),
    }


def slot_categories(metadata: dict, num_actors: int) -> tuple[list[str], list[str]]:
    """Track token and nuPlan category per compact dynamic slot."""
    tokens, categories = [""] * num_actors, [""] * num_actors
    for actor in metadata["actors"]:
        if actor.get("storage", "dynamic") != "dynamic" or "slot" not in actor:
            continue
        slot = int(actor["slot"])
        if slot >= num_actors:
            raise ValueError(f"Metadata slot {slot} exceeds the scene's {num_actors} actor slots")
        tokens[slot] = actor["track_token"]
        category = actor.get("category") or actor.get("source_type") or ""
        categories[slot] = "ego" if actor["track_token"] == "ego" else category.lower()
    if any(not token for token in tokens):
        raise ValueError("Metadata does not name every dynamic slot")
    return tokens, categories


def make_scenario(
    database: str | Path,
    token: str,
    map_name: str,
    *,
    scenario_type: str = "unknown",
    timestamp: int | None = None,
    data_root: str | None = None,
    maps_root: str | None = None,
    duration: float = 45.2,
):
    """Construct the 10 Hz NuPlanScenario the exporter samples from (45.2 s by default).

    Metric validation passes the episode length, as nuPlan's 15 s simulations do.
    """
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
        ScenarioExtractionInfo,
    )

    database = Path(database)
    data_root = data_root or os.environ.get("NUPLAN_DATA_ROOT", DEFAULT_DATA_ROOT)
    data_root = os.path.expanduser(data_root)
    maps_root = maps_root or os.environ.get("NUPLAN_MAPS_ROOT", f"{data_root}/maps")
    if not database.is_file():
        raise ValueError(f"Source database does not exist: {database}")
    if timestamp is None:
        with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as conn:
            row = conn.execute(
                "SELECT timestamp FROM lidar_pc WHERE token=?", (bytes.fromhex(token),)
            ).fetchone()
        if row is None:
            raise ValueError(f"Token {token} is not in {database}")
        timestamp = int(row[0])
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=str(database),
        initial_lidar_token=token,
        initial_lidar_timestamp=int(timestamp),
        scenario_type=scenario_type,
        map_root=maps_root,
        map_version=os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
        map_name=map_name,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_duration=duration, subsample_ratio=0.5
        ),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


def build_scenario(metadata: dict, **roots):
    """Rebuild the scenario recorded in an exported scene's metadata."""
    fields = source_fields(metadata)
    return make_scenario(
        fields["database"],
        fields["token"],
        fields["map_name"],
        scenario_type=fields["scenario_type"],
        timestamp=fields["timestamp"],
        **roots,
    )
