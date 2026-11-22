"""Case metadata and fixed-size recorded-frame access."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from cs5478_project.server.schemas import CaseDetail, CaseList, FrameChunk
from cs5478_project.server.services.case_store import CaseStore
from cs5478_project.visualization.map import SceneMap

router = APIRouter(prefix="/api/v1")


def get_store(request: Request) -> CaseStore:
    return request.app.state.case_store


Store = Annotated[CaseStore, Depends(get_store)]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/cases", response_model=CaseList)
def list_cases(store: Store) -> CaseList:
    return CaseList(cases=store.list_cases())


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str, store: Store) -> CaseDetail:
    try:
        return store.detail(case_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown case ID") from error


@router.get("/cases/{case_id}/map", response_model=SceneMap)
def get_map(case_id: str, store: Store) -> SceneMap:
    try:
        return store.scene_map(case_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown case ID") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/cases/{case_id}/frames", response_model=FrameChunk)
def get_frames(
    case_id: str,
    store: Store,
    start: Annotated[int, Query(ge=0)] = 0,
    count: Annotated[int, Query(ge=1, le=256)] = 128,
) -> FrameChunk:
    try:
        return store.frames(case_id, start, count)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown case ID") from error
    except IndexError as error:
        raise HTTPException(status_code=416, detail=str(error)) from error
