"""Replay acceptance checks independent of learned dynamics and nuPlan parsing."""

import pytest
import torch

from cs5478_project.agents.replay import LogAgent
from cs5478_project.data.dataloader import SceneDataLoader
from cs5478_project.data.dataset import SceneDataset
from cs5478_project.data.synthetic import make_synthetic_scene
from cs5478_project.datatypes.scene import SceneBatch
from cs5478_project.datatypes.state import WorldState, clone_state
from cs5478_project.env.env import ReplayEnv
from cs5478_project.env.step import step_kernel


def make_batch(*frame_counts: int, max_frames: int = 8, max_actors: int = 5) -> SceneBatch:
    scenes = [
        make_synthetic_scene(scenario_id=f"world-{i}", num_frames=frames, num_actors=3)
        for i, frames in enumerate(frame_counts)
    ]
    return next(
        iter(
            SceneDataLoader(
                SceneDataset(scenes),
                batch_size=len(scenes),
                max_actors=max_actors,
                max_frames=max_frames,
            )
        )
    ).tensors


def assert_state_equal(actual: WorldState, expected: WorldState) -> None:
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_log_agent_selects_all_recorded_fields_per_world() -> None:
    batch = make_batch(6, 5)
    reference_state = WorldState(
        *(
            torch.arange(field.numel(), dtype=torch.float32).reshape(field.shape) + offset * 1000
            for offset, field in enumerate(batch.reference_state)
        )
    )
    index = torch.tensor([4, 2])
    result = LogAgent.get_state(reference_state, index)
    for actual, reference in zip(result, reference_state, strict=True):
        torch.testing.assert_close(actual[0], reference[0, 4], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], reference[1, 2], rtol=0, atol=0)


def test_reset_copies_initial_state_and_has_no_transition() -> None:
    batch = make_batch(5, 1)
    env = ReplayEnv()
    result = env.reset(batch)
    assert_state_equal(result.state, batch.initial_state)
    assert result.episode.step.tolist() == [0, 0]
    assert result.episode.done.tolist() == [False, True]
    assert not result.advanced.any()
    assert not result.just_finished.any()
    original = batch.initial_state.position.clone()
    result.state.position.add_(10)
    torch.testing.assert_close(batch.initial_state.position, original, rtol=0, atol=0)


def test_full_playback_includes_final_frame_and_freezes_completed_worlds() -> None:
    batch = make_batch(6, 3, 1)
    env = ReplayEnv()
    env.reset(batch)
    finish_count = torch.zeros(3, dtype=torch.int64)
    for tick in range(1, 9):
        result = env.step()
        finish_count += result.just_finished
        assert result.episode.step.tolist() == [min(tick, 5), min(tick, 2), 0]
        assert result.advanced.tolist() == [tick <= 5, tick <= 2, False]
        assert result.episode.done.tolist() == [tick >= 5, tick >= 2, True]
        for world, final_frame in enumerate((5, 2, 0)):
            for actual, reference in zip(result.state, batch.reference_state, strict=True):
                valid = batch.actor_mask[world]
                torch.testing.assert_close(
                    actual[world, valid],
                    reference[world, min(tick, final_frame), valid],
                    rtol=0,
                    atol=0,
                )
    assert finish_count.tolist() == [1, 1, 0]


def test_padding_is_held_even_when_recorded_padding_changes() -> None:
    batch = make_batch(4)
    initial = clone_state(batch.initial_state)
    reference = clone_state(batch.reference_state)
    for field in initial:
        field[:, 3:] = 11.0
    for field in reference:
        field[:, :, 3:] = 22.0
    batch = batch._replace(initial_state=initial, reference_state=reference)
    env = ReplayEnv()
    env.reset(batch)
    for _ in range(5):
        result = env.step()
        for actual, expected in zip(result.state, initial, strict=True):
            torch.testing.assert_close(actual[:, 3:], expected[:, 3:], rtol=0, atol=0)


def test_partial_reset_replaces_scene_and_state_only_for_selected_worlds() -> None:
    original = make_batch(6, 3)
    incoming = make_batch(2, 7)
    incoming.initial_state.position.add_(100)
    incoming.reference_state.position.add_(100)
    env = ReplayEnv()
    env.reset(original)
    env.step()
    previous = env.step()
    held_state = clone_state(previous.state)
    result = env.reset(incoming, mask=torch.tensor([True, False]))
    assert result.episode.step.tolist() == [0, 2]
    assert result.episode.done.tolist() == [False, True]
    assert env.scene.num_frames.tolist() == [2, 3]
    for actual, new, held in zip(result.state, incoming.initial_state, held_state, strict=True):
        torch.testing.assert_close(actual[0], new[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], held[1], rtol=0, atol=0)
    for actual, old in zip(env.scene.reference_state, original.reference_state, strict=True):
        torch.testing.assert_close(actual[1], old[1], rtol=0, atol=0)
    result = env.step()
    assert result.episode.step.tolist() == [1, 2]
    assert result.just_finished.tolist() == [True, False]
    for actual, reference in zip(result.state, incoming.reference_state, strict=True):
        torch.testing.assert_close(actual[0], reference[0, 1], rtol=0, atol=0)
    env.reset(incoming, mask=torch.tensor([False, False]))
    assert env.episode.step.tolist() == [1, 2]
    assert env.scene.num_frames.tolist() == [2, 3]


def test_reset_rejects_capacity_changes_and_bad_masks() -> None:
    env = ReplayEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step()
    batch = make_batch(4, 3)
    with pytest.raises(ValueError, match="first reset"):
        env.reset(batch, mask=torch.tensor([True, True]))
    env.reset(batch)
    for bad_mask in (torch.ones(2), torch.tensor([True]), torch.ones(2, 1, dtype=torch.bool)):
        with pytest.raises(ValueError, match="Reset mask"):
            env.reset(batch, mask=bad_mask)
    with pytest.raises(ValueError, match="preserve"):
        env.reset(make_batch(4, 3, max_frames=9))


def test_compiled_step_matches_eager_without_recompiling_for_content_or_counters() -> None:
    captured_graphs = []

    def counting_backend(graph, example_inputs):
        captured_graphs.append(graph)
        return graph.forward

    compiled = torch.compile(step_kernel, fullgraph=True, dynamic=False, backend=counting_backend)
    env = ReplayEnv()
    first_batch = make_batch(6, 3)
    changed_batch = make_batch(2, 7)
    changed_batch.actor_mask[0, 2] = False
    changed_batch.reference_valid[0, :, 2] = False
    changed_batch.routes.mask[0, 2] = False
    changed_batch.routes.source[0, 2] = 0
    changed_batch.routes.horizon_s[0, 2] = 0
    changed_batch.routes.start_time_s[0, 2] = 0
    for batch in (first_batch, changed_batch):
        result = env.reset(batch)
        state, episode = result.state, result.episode
        for _ in range(9):
            expected = step_kernel(state, episode, batch)
            actual = compiled(state, episode, batch)
            assert_state_equal(actual.state, expected.state)
            for actual_flag, expected_flag in zip(actual.episode, expected.episode, strict=True):
                torch.testing.assert_close(actual_flag, expected_flag, rtol=0, atol=0)
            assert torch.equal(actual.advanced, expected.advanced)
            assert torch.equal(actual.just_finished, expected.just_finished)
            state, episode = actual.state, actual.episode
    assert len(captured_graphs) == 1
