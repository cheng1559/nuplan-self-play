"""Serialize static geometry once with half-open observation intervals."""

import numpy as np

from cs5478_project.server.schemas import Actor


def static_actor_metadata(objects: dict[str, np.ndarray] | None) -> list[Actor]:
    if objects is None:
        return []
    actors = []
    for slot in np.flatnonzero(objects["mask"]):
        edges = np.flatnonzero(np.diff(np.r_[False, objects["valid"][:, slot], False]))
        actors.append(
            Actor(
                id=f"actor-{objects['ids'][slot]}",
                slot=int(slot),
                type="static",
                control_type="static",
                length=float(objects["size"][slot, 0]),
                width=float(objects["size"][slot, 1]),
                reference_to_center=0.0,
                static_position=objects["position"][slot].tolist(),
                static_heading=float(objects["heading"][slot]),
                presence_intervals=edges.reshape(-1, 2).tolist(),
            )
        )
    return actors
