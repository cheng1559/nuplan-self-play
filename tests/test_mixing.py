"""World modes for mixed training: log replay, reactive, and self-play controllers."""

import torch
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.datatypes.control import ControlType
from cs5478_project.datatypes.scene import ActorType
from cs5478_project.training.mixing import (
    WorldMode,
    mixed_controls,
    sample_modes,
    self_play_candidates,
)

V, P = ActorType.VEHICLE, ActorType.PEDESTRIAN


def scene():
    # Ego at the origin; vehicles 20 m and 40 m ahead, one 60 m ahead overlapping a
    # fourth; one pedestrian on the side.
    return traffic_scene(
        ((0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (60.0, 0.0), (61.0, 0.0), (10.0, 8.0)),
        (5.0, 5.0, 5.0, 5.0, 5.0, 1.0),
        (V, V, V, V, V, P),
        frames=11,
    )


def test_self_play_takes_the_nearest_clear_vehicles():
    batch = batch_of(scene(), capacity=8)
    picked = self_play_candidates(batch, 3)[0, :6].tolist()
    # Ego is never a candidate; the overlapping pair at 60 m is skipped.
    assert picked == [False, True, True, False, False, False]
    assert self_play_candidates(batch, 1)[0, :6].tolist() == [False, True] + [False] * 4


def test_world_modes_assign_controllers():
    batch = batch_of(scene(), scene(), scene(), capacity=8)
    modes = torch.tensor([WorldMode.LOG, WorldMode.REACTIVE, WorldMode.SELF_PLAY])
    controls = mixed_controls(batch, modes, self_play_agents=1)[:, :6]
    railed = batch.rails.rail_valid[:, :6]
    ego, log, idm = ControlType.SINGLE_FRAME, ControlType.LOG, ControlType.IDM
    assert controls[0].tolist() == [ego] + [log] * 5
    reactive = controls[1]
    assert reactive[0] == ego and reactive[5] == ControlType.PEDESTRIAN
    assert all(int(reactive[i]) == (idm if railed[1, i] else log) for i in range(1, 5))
    self_play = controls[2]
    assert self_play[0] == ego and self_play[1] == ego  # the nearest clear vehicle
    assert self_play[5] == ControlType.PEDESTRIAN
    no_rail = ControlType.ABSENT
    assert all(int(self_play[i]) == (idm if railed[2, i] else no_rail) for i in range(2, 5))
    assert not (controls[2] == log).any()
    # Padding stays LOG.
    assert (mixed_controls(batch, modes)[:, 6:] == log).all()


def test_mode_weights():
    modes = sample_modes(3000, (1.0, 0.0, 3.0), "cpu", torch.Generator().manual_seed(0))
    assert not (modes == WorldMode.REACTIVE).any()
    assert 0.7 < float((modes == WorldMode.SELF_PLAY).float().mean()) < 0.8
