"""Per-actor controller assignments kept separate from physical actor categories."""

from enum import IntEnum


class ControlType(IntEnum):
    """Integer values for fixed-shape [B, A] controller tensors."""

    LOG = 0
    IDM = 1
    PEDESTRIAN = 2
    # 3 is unused (kept so stored controller codes stay stable).
    SINGLE_FRAME = 4
    # Not simulated at all, like vehicles nuPlan's reactive agents never create.
    ABSENT = 5
