"""Checkpoint control semantics: measured acceleration, policy hold, and recovery."""

import torch
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.datatypes.action import VehicleAction
from cs5478_project.dynamics.action_adapter import step_jerk_bicycle
from cs5478_project.dynamics.bicycle import heading_vector
from cs5478_project.models.config import SingleFrameConfig


def test_zero_jerk_uses_actual_acceleration_instead_of_stale_command():
    batch = batch_of(traffic_scene())
    state = batch.initial_state._replace(
        acceleration=torch.ones_like(batch.initial_state.acceleration),
        acceleration_command=torch.full_like(batch.initial_state.acceleration, -6),
    )
    zeros = torch.zeros_like(state.heading)
    result = step_jerk_bicycle(
        state,
        VehicleAction(zeros, zeros),
        batch.wheelbase,
        batch.dt[:, None],
        SingleFrameConfig(),
        torch.ones_like(batch.actor_mask),
    )
    torch.testing.assert_close(result.acceleration_command, state.acceleration, atol=0, rtol=0)
    torch.testing.assert_close(result.acceleration, state.acceleration, atol=1e-5, rtol=0)


def test_policy_target_is_held_and_positive_jerk_obeys_checkpoint_limit():
    batch = batch_of(traffic_scene())
    state = batch.initial_state._replace(
        acceleration=torch.ones_like(batch.initial_state.acceleration)
    )
    action = VehicleAction(torch.full_like(state.heading, 5), torch.zeros_like(state.heading))
    # A 0.2 s policy interval: target = 1 + min(5, positive limit 1) * 0.2 = 1.2.
    config = SingleFrameConfig(policy_interval_s=0.2)
    first = step_jerk_bicycle(
        state,
        action,
        batch.wheelbase,
        batch.dt[:, None],
        config,
        torch.ones_like(batch.actor_mask),
    )
    second = step_jerk_bicycle(
        first,
        action,
        batch.wheelbase,
        batch.dt[:, None],
        config,
        torch.zeros_like(batch.actor_mask),
    )
    torch.testing.assert_close(first.acceleration_command, torch.full_like(state.heading, 1.2))
    torch.testing.assert_close(
        second.acceleration_command, first.acceleration_command, atol=0, rtol=0
    )
    assert (second.acceleration > first.acceleration).all()
    assert (second.acceleration < second.acceleration_command).all()


def test_braking_target_does_not_block_acceleration_from_rest():
    batch = batch_of(traffic_scene())
    state = batch.initial_state._replace(
        velocity=torch.zeros_like(batch.initial_state.velocity),
        acceleration=torch.zeros_like(batch.initial_state.acceleration),
        acceleration_command=torch.full_like(batch.initial_state.acceleration, -4),
    )
    action = VehicleAction(torch.full_like(state.heading, 5), torch.zeros_like(state.heading))
    result = step_jerk_bicycle(
        state,
        action,
        batch.wheelbase,
        batch.dt[:, None],
        SingleFrameConfig(),
        torch.ones_like(batch.actor_mask),
    )
    assert (result.velocity.norm(dim=-1) > 0).all()
    assert (result.acceleration_command > 0).all()


def test_hard_braking_stops_exactly_and_never_reverses():
    batch = batch_of(traffic_scene(speeds=(1.0,)))
    state = batch.initial_state._replace(
        acceleration=torch.full_like(batch.initial_state.acceleration, -3.0)
    )
    brake = VehicleAction(torch.full_like(state.heading, -8.0), torch.zeros_like(state.heading))
    speeds = []
    for step in range(40):
        state = step_jerk_bicycle(
            state,
            brake,
            batch.wheelbase,
            batch.dt[:, None],
            SingleFrameConfig(),
            torch.full_like(batch.actor_mask, step % 2 == 0),
        )
        speeds.append(float((state.velocity * heading_vector(state.heading)).sum(-1)[0, 0]))
    assert min(speeds) >= -1e-6
    assert abs(speeds[-1]) < 1e-6 and abs(float(state.acceleration[0, 0])) < 1e-4
