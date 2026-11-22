"""Read static scene-local polylines independently of simulator tensors."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from cs5478_project.datatypes.map import MapData

Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False)]

MapPolylineKind = Literal[
    "lane_centerline",
    "lane_connector_centerline",
    "lane_boundary",
    "road_boundary",
    "crosswalk",
    "stop_line",
]


class _MapModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class WorldFromScene(_MapModel):
    x: Coordinate
    y: Coordinate
    heading: Coordinate


class MapPolyline(_MapModel):
    id: Annotated[str, Field(min_length=1)]
    kind: MapPolylineKind
    points: Annotated[list[tuple[Coordinate, Coordinate]], Field(min_length=2)]


class SceneMap(_MapModel):
    """Static display geometry in the same local frame as recorded actors."""

    format: Literal["scene_map"]
    schema_version: Literal[1]
    scenario_id: Annotated[str, Field(min_length=1)]
    map_name: Annotated[str, Field(min_length=1)]
    coordinate_frame: Literal["scene"]
    world_from_scene: WorldFromScene
    bounds: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    polylines: list[MapPolyline]

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        xmin, ymin, xmax, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("bounds must be ordered [xmin, ymin, xmax, ymax]")
        identifiers = [line.id for line in self.polylines]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("polyline IDs must be unique")
        return self


def load_scene_map(path: Path) -> SceneMap:
    """Reject malformed sidecars with the source path in the error message."""
    try:
        return SceneMap.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid scene map {path}: {error}") from error


def map_data_from_display(display: SceneMap) -> "MapData":
    """Convert display polylines to the scene's CPU segment representation."""
    import torch

    from cs5478_project.datatypes.map import MapData, MapElementType

    segments, types, indices = [], [], []
    for index, line in enumerate(display.polylines):
        for start, end in zip(line.points[:-1], line.points[1:], strict=True):
            segments.append((start, end))
            types.append(int(MapElementType[line.kind.upper()]))
            indices.append(index)
    data = MapData(
        map_name=display.map_name,
        world_from_scene=(
            display.world_from_scene.x,
            display.world_from_scene.y,
            display.world_from_scene.heading,
        ),
        bounds=display.bounds,
        segments=torch.tensor(segments, dtype=torch.float32).reshape(-1, 2, 2),
        segment_type=torch.tensor(types, dtype=torch.int64),
        polyline_index=torch.tensor(indices, dtype=torch.int64),
        polyline_ids=tuple(line.id for line in display.polylines),
    )
    data.validate()
    from cs5478_project.data.map_chunks import with_chunks

    return with_chunks(data)


def display_map_from_data(data: "MapData", scenario_id: str) -> SceneMap:
    """Reconstruct display polylines from the same segments used by simulation."""
    from cs5478_project.datatypes.map import MapElementType

    data.validate()
    segments = data.segments.tolist()
    types = data.segment_type.tolist()
    indices = data.polyline_index.tolist()
    points: list[list[tuple[float, float]]] = [[] for _ in data.polyline_ids]
    kinds: list[str | None] = [None for _ in data.polyline_ids]
    for segment, kind, index in zip(segments, types, indices, strict=True):
        start, end = (tuple(point) for point in segment)
        name = MapElementType(kind).name.lower()
        if not points[index]:
            points[index].append(start)
            kinds[index] = name
        elif points[index][-1] != start or kinds[index] != name:
            raise ValueError("Map polyline segments must be connected and have one type")
        points[index].append(end)
    x, y, heading = data.world_from_scene
    return SceneMap(
        format="scene_map",
        schema_version=1,
        scenario_id=scenario_id,
        map_name=data.map_name,
        coordinate_frame="scene",
        world_from_scene=WorldFromScene(x=x, y=y, heading=heading),
        bounds=data.bounds,
        polylines=[
            MapPolyline(id=identifier, kind=kind, points=vertices)
            for identifier, kind, vertices in zip(data.polyline_ids, kinds, points, strict=True)
        ],
    )
