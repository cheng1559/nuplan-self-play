"""Small shared map geometry for scene and environment contract tests."""

import pytest
import torch

from cs5478_project.datatypes import MapData, MapElementType


@pytest.fixture
def map_data():
    """Two connected centerline segments and one independent road boundary."""
    return MapData(
        map_name="test-map",
        world_from_scene=(100.0, 200.0, 0.5),
        bounds=(-10.0, -10.0, 20.0, 20.0),
        segments=torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [4.0, 1.0]], [[0.0, 5.0], [5.0, 5.0]]]
        ),
        segment_type=torch.tensor(
            [
                MapElementType.LANE_CENTERLINE,
                MapElementType.LANE_CENTERLINE,
                MapElementType.ROAD_BOUNDARY,
            ]
        ),
        polyline_index=torch.tensor([0, 0, 1]),
        polyline_ids=("lane-1", "road-1"),
    )
