"""Compare real Inductor CPU/CUDA replay against eager tensor execution."""

from dataclasses import replace

import pytest
import torch

from cs5478_project.data import SceneDataLoader, SceneDataset, make_synthetic_scene
from cs5478_project.env import ReplayEnv


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inductor_replays_every_field_and_freezes_completed_worlds(device, map_data):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dataset = SceneDataset(
        [
            replace(make_synthetic_scene(scenario_id="long", num_frames=5), map=map_data),
            make_synthetic_scene(scenario_id="short", num_frames=2, num_actors=1),
        ]
    )
    batch = next(
        iter(SceneDataLoader(dataset, batch_size=2, max_actors=5, max_frames=7, device=device))
    ).tensors
    eager, compiled = ReplayEnv(), ReplayEnv(compile_step=True)
    expected, actual = eager.reset(batch), compiled.reset(batch)
    assert compiled.scene.map.segment_mask.sum(dim=1).tolist() == [3, 0]
    assert compiled.scene.map.segments.device.type == device
    for tick in range(7):
        if tick == 3:
            mask = torch.tensor([False, True], device=device)
            expected = eager.reset(batch, mask=mask)
            actual = compiled.reset(batch, mask=mask)
        for expected_field, actual_field in zip(expected.state, actual.state, strict=True):
            torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)
        assert torch.equal(actual.episode.step, expected.episode.step)
        assert torch.equal(actual.episode.done, expected.episode.done)
        assert torch.equal(actual.advanced, expected.advanced)
        assert torch.equal(actual.just_finished, expected.just_finished)
        expected, actual = eager.step(), compiled.step()
