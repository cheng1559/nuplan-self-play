"""Build completed-batch simulation and archived artifact APIs."""

from pathlib import Path

from fastapi import FastAPI

from cs5478_project.defaults import EPISODE_FRAMES, MAX_DYNAMIC_ACTORS, MAX_STATIC_OBJECTS
from cs5478_project.server.routers.cases import router
from cs5478_project.server.routers.rollouts import router as rollout_router
from cs5478_project.server.schemas import RolloutSettings
from cs5478_project.server.services.case_store import CaseStore
from cs5478_project.server.services.rollout_service import RolloutService


def create_app(
    *,
    artifact_roots: list[Path] | None = None,
    include_demo: bool = False,
    device: str = "auto",
    compile_step: bool = False,
    max_actors: int = MAX_DYNAMIC_ACTORS,
    max_static_objects: int = MAX_STATIC_OBJECTS,
    max_frames: int = EPISODE_FRAMES,
    max_map_segments: int = 2048,
    max_batch_size: int = 256,
) -> FastAPI:
    """Create isolated input catalogs and a serialized live batch executor."""
    app = FastAPI(title="CS5478 Simulation Visualizer", version="0.2.0")
    app.state.case_store = CaseStore(artifact_roots, include_demo=include_demo)
    app.state.rollout_service = RolloutService(
        artifact_roots,
        include_demo=include_demo,
        settings=RolloutSettings(
            device=device,
            compiled=compile_step,
            max_actors=max_actors,
            max_static_objects=max_static_objects,
            max_frames=max_frames,
            max_map_segments=max_map_segments,
            max_batch_size=max_batch_size,
        ),
    )
    app.include_router(router)
    app.include_router(rollout_router)
    return app
