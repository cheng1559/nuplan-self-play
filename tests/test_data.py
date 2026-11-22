"""Acceptance checks for normalized scene input and fixed-capacity loading."""

from dataclasses import replace

import pytest
import torch

from cs5478_project.data import (
    SceneDataLoader,
    SceneDataset,
    load_scene,
    make_synthetic_scene,
    save_scene,
)
from cs5478_project.datatypes import WorldState, validate_scene_batch


def test_scene_roundtrip_preserves_every_recorded_field(tmp_path):
    scene = make_synthetic_scene(scenario_id="roundtrip", num_frames=7, num_actors=4)
    path = tmp_path / "nested" / "scene.pt"
    save_scene(scene, path)
    restored = load_scene(path)
    assert restored.scenario_id == scene.scenario_id
    assert restored.dt == scene.dt
    for name in ("actor_type", "actor_mask", "size", "wheelbase", "reference_to_center"):
        assert torch.equal(getattr(restored, name), getattr(scene, name))
    for original, loaded in zip(
        scene.initial_state + scene.reference_state,
        restored.initial_state + restored.reference_state,
        strict=True,
    ):
        assert torch.equal(original, loaded)
    assert torch.equal(scene.reference_valid, restored.reference_valid)


def test_fixed_capacities_preserve_data_and_pad_final_frame():
    short = make_synthetic_scene(scenario_id="short", num_frames=3, num_actors=1)
    long = make_synthetic_scene(scenario_id="long", num_frames=6, num_actors=3)
    loaded = next(
        iter(SceneDataLoader(SceneDataset([short, long]), batch_size=2, max_actors=5, max_frames=8))
    )
    batch = loaded.tensors
    validate_scene_batch(batch)
    assert loaded.scenario_ids == ("short", "long")
    assert batch.reference_state.position.shape == (2, 8, 5, 2)
    assert batch.initial_state.position.shape == (2, 5, 2)
    assert batch.num_frames.tolist() == [3, 6]
    assert batch.actor_mask.sum(dim=1).tolist() == [1, 2]
    assert not batch.reference_valid[0, 3:].any()
    assert not batch.reference_valid[1, 6:].any()
    for actual, expected in zip(batch.reference_state, short.reference_state, strict=True):
        assert torch.equal(actual[0, :3, :1], expected)
        assert torch.equal(actual[0, 3:, :1], expected[-1:].expand_as(actual[0, 3:, :1]))
        assert torch.isfinite(actual).all()
    assert torch.all(batch.wheelbase > 0)
    assert torch.all(batch.size > 0)


def test_dataset_accepts_in_memory_and_file_inputs(tmp_path):
    scene = make_synthetic_scene(num_frames=2)
    path = tmp_path / "scene.pt"
    save_scene(scene, path)
    dataset = SceneDataset([scene, path, str(path)])
    assert len(dataset) == 3
    assert dataset[0] is scene
    assert dataset[1].scenario_id == scene.scenario_id
    assert dataset[2].scenario_id == scene.scenario_id


@pytest.mark.parametrize(
    ("max_actors", "max_frames", "message"),
    [
        (1, 10, "max_actors"),
        (3, 4, "max_frames"),
    ],
)
def test_capacity_overflow_is_rejected(max_actors, max_frames, message):
    loader = SceneDataLoader(
        SceneDataset([make_synthetic_scene(num_frames=5, num_actors=3)]),
        batch_size=1,
        max_actors=max_actors,
        max_frames=max_frames,
    )
    with pytest.raises(ValueError, match=message):
        next(iter(loader))


def test_incomplete_batch_cannot_change_compiled_batch_shape():
    dataset = SceneDataset([make_synthetic_scene(num_frames=2)] * 3)
    with pytest.raises(ValueError, match="divisible"):
        SceneDataLoader(dataset, batch_size=2, max_actors=3, max_frames=2, drop_last=False)
    loader = SceneDataLoader(dataset, batch_size=2, max_actors=3, max_frames=2)
    batches = list(loader)
    assert len(loader) == len(batches) == 1
    assert batches[0].tensors.actor_mask.shape == (2, 3)


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_time_step_is_rejected(dt):
    scene = replace(make_synthetic_scene(num_frames=2), dt=dt)
    with pytest.raises(ValueError, match="dt"):
        scene.validate()


def test_missing_reference_state_is_rejected_instead_of_despawning_actor():
    scene = make_synthetic_scene(num_frames=3)
    scene.reference_valid[1, 0] = False
    with pytest.raises(ValueError, match="ego must be present"):
        scene.validate()


def test_initial_state_must_match_frame_zero():
    scene = make_synthetic_scene(num_frames=3)
    scene.initial_state.position[0, 0] += 1
    with pytest.raises(ValueError, match="frame zero"):
        scene.validate()


def test_nonfinite_padding_is_rejected():
    scene = make_synthetic_scene(num_frames=3)
    scene.actor_mask[1] = False
    scene.reference_valid[:, 1] = False
    scene.reference_state.heading[1, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        scene.validate()


def test_wrong_state_dtype_is_rejected():
    scene = make_synthetic_scene(num_frames=3)
    changed = scene.reference_state._replace(velocity=scene.reference_state.velocity.double())
    with pytest.raises(ValueError, match="dtype"):
        replace(scene, reference_state=changed).validate()


def test_one_frame_scene_has_a_valid_initial_state():
    scene = make_synthetic_scene(num_frames=1, num_actors=1)
    loaded = next(
        iter(SceneDataLoader(SceneDataset([scene]), batch_size=1, max_actors=2, max_frames=3))
    )
    validate_scene_batch(loaded.tensors)
    assert loaded.tensors.num_frames.tolist() == [1]
    assert loaded.tensors.reference_valid.sum() == 1


def test_batch_validation_checks_counts_and_validity():
    batch = next(
        iter(
            SceneDataLoader(
                SceneDataset([make_synthetic_scene(num_frames=2)]),
                batch_size=1,
                max_actors=3,
                max_frames=4,
            )
        )
    ).tensors
    with pytest.raises(ValueError, match="num_frames"):
        validate_scene_batch(batch._replace(num_frames=torch.tensor([5])))
    with pytest.raises(ValueError, match="frame counts"):
        validate_scene_batch(batch._replace(reference_valid=torch.ones_like(batch.reference_valid)))


def test_file_format_version_is_checked(tmp_path):
    path = tmp_path / "unsupported.pt"
    torch.save({"format_version": -1}, path)
    with pytest.raises(ValueError, match="Unsupported scene format"):
        load_scene(path)


def test_synthetic_records_include_distinct_motion_and_physical_types():
    scene = make_synthetic_scene(num_frames=9)
    assert scene.actor_type.tolist() == [0, 1]
    assert scene.reference_state.heading[-1, 0] > scene.reference_state.heading[0, 0]
    assert not torch.equal(scene.reference_state.position[-1, 1], scene.initial_state.position[1])
    assert scene.static.position.shape == (1, 2)
    assert scene.static.valid.all()
    assert scene.static.ids.tolist() == [2]
    assert isinstance(scene.reference_state, WorldState)
