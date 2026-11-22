"""Blindness is a query-local observation mask shared by all reactive controllers."""

import math

import pytest
import torch
from test_rule_agents import batch_of, traffic_scene

from cs5478_project.datatypes.control import ControlType
from cs5478_project.env import RuleBasedEnv


@pytest.mark.parametrize("kind", [ControlType.IDM, ControlType.PEDESTRIAN])
def test_blind_controller_ignores_vehicle_conflict(kind):
    if kind == ControlType.IDM:
        scene = traffic_scene(((0.0, 0.0), (6.0, 0.0)), (4.0, 0.0), (0, 0), frames=8)
    else:
        scene = traffic_scene(
            ((0.0, 0.0), (2.0, -2.0)),
            (1.4, 2.0),
            (1, 0),
            headings=(0.0, math.pi / 2),
            frames=8,
        )
    batch = batch_of(scene)
    modes = torch.zeros_like(batch.actor_type)
    modes[:, 0] = kind
    sighted, blind = RuleBasedEnv(), RuleBasedEnv()
    flags = torch.zeros_like(batch.actor_mask)
    flags[:, 0] = True
    sighted.reset(batch, control_types=modes)
    blind.reset(batch, control_types=modes, blind=flags)
    before = sighted.step()
    after = blind.step()
    assert after.state.velocity[0, 0].norm() > before.state.velocity[0, 0].norm()
    torch.testing.assert_close(after.state.position[:, 1], before.state.position[:, 1])


def test_blind_partial_reset_and_validation():
    scene = traffic_scene(frames=8)
    batch = batch_of(scene, scene)
    env = RuleBasedEnv()
    flags = batch.actor_mask.clone()
    env.reset(batch, blind=flags)
    env.reset(batch, mask=torch.tensor([True, False]))
    assert not env.blind[0].any()
    assert torch.equal(env.blind[1], flags[1])
    with pytest.raises(ValueError, match="bool"):
        env.reset(batch, blind=flags.long())
    with pytest.raises(ValueError, match="Padding"):
        env.reset(batch, blind=torch.ones_like(flags))


def test_fullgraph_blind_values_do_not_recompile():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    batch = batch_of(traffic_scene(((0.0, 0.0), (6.0, 0.0)), (4.0, 0.0), (0, 0), frames=8))
    env = RuleBasedEnv()
    # Reset picks the step kernel for the controllers in use; compile each one once.
    eager_kernel, kernels = env._kernel, {}

    def kernel(**flags):
        key = tuple(flags.items())
        if key not in kernels:
            kernels[key] = torch.compile(
                eager_kernel(**flags), fullgraph=True, dynamic=False, backend=backend
            )
        return kernels[key]

    env._kernel = kernel
    for enabled in (False, True, False):
        flags = torch.zeros_like(batch.actor_mask)
        flags[0, 0] = enabled
        env.reset(batch, blind=flags)
        expected = RuleBasedEnv()
        expected.reset(batch, blind=flags)
        for _ in range(3):
            actual, reference = env.step(), expected.step()
            for a, b in zip(actual.state, reference.state, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert len(graphs) == 1
