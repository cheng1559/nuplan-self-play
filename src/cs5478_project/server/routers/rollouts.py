"""Execute complete batches before sending their states and static maps."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from cs5478_project.server.schemas import (
    BatchRollout,
    BatchRolloutRequest,
    ObservationTrack,
    PolicyTrack,
    RolloutSourceList,
    RolloutWorld,
)
from cs5478_project.server.services.rollout_service import (
    RolloutBusyError,
    RolloutService,
    RolloutUnavailableError,
    UnknownRollout,
    UnknownRolloutSource,
)

router = APIRouter(prefix="/api/v1")


def get_rollout_service(request: Request) -> RolloutService:
    return request.app.state.rollout_service


Service = Annotated[RolloutService, Depends(get_rollout_service)]


@router.get("/rollout-sources", response_model=RolloutSourceList)
def list_sources(service: Service) -> RolloutSourceList:
    return service.list_sources()


@router.post("/rollouts", response_model=BatchRollout)
def run_batch(payload: BatchRolloutRequest, service: Service) -> BatchRollout:
    try:
        return service.run(
            payload.source_ids,
            preset=payload.preset,
            agent_controls=payload.agent_controls,
            added_actors=payload.added_actors,
            checkpoint=payload.checkpoint,
            checkpoint_config=payload.checkpoint_config,
            sampling_method=payload.sampling_method,
            lazy=payload.lazy,
            goal_horizon_s=payload.goal_horizon_s,
        )
    except UnknownRolloutSource as error:
        raise HTTPException(
            status_code=404, detail=f"Unknown rollout source: {error.args[0]}"
        ) from error
    except RolloutBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RolloutUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/rollouts/{run_id}/worlds/{world}", response_model=RolloutWorld)
def rollout_world(run_id: str, world: int, service: Service):
    try:
        return service.world(run_id, world)
    except UnknownRollout as error:
        raise HTTPException(
            status_code=404, detail=f"Rollout {error.args[0]} is not the latest run"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/rollouts/{run_id}/policy", response_model=PolicyTrack)
def policy(run_id: str, world: int, actor_id: str, service: Service):
    try:
        return service.policy(run_id, world, actor_id)
    except UnknownRollout as error:
        raise HTTPException(
            status_code=404, detail=f"Rollout {error.args[0]} is not the latest run"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/rollouts/{run_id}/observations", response_model=ObservationTrack)
def observations(run_id: str, world: int, actor_id: str, service: Service):
    try:
        return service.observations(run_id, world, actor_id)
    except UnknownRollout as error:
        raise HTTPException(
            status_code=404, detail=f"Rollout {error.args[0]} is not the latest SingleFrameNet run"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
