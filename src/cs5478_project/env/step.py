"""Fixed-shape log playback suitable for full-graph compilation."""

import torch

from cs5478_project.agents.replay import LogAgent
from cs5478_project.datatypes.scene import SceneBatch
from cs5478_project.datatypes.state import EpisodeState, WorldState, select_state
from cs5478_project.datatypes.transition import StepResult


def step_kernel(state: WorldState, episode: EpisodeState, scene: SceneBatch) -> StepResult:
    """Advance unfinished worlds once and hold completed worlds at their final frame."""
    advanced = ~episode.done
    next_step = torch.minimum(episode.step + advanced.to(torch.int64), scene.num_frames - 1)
    recorded_state = LogAgent.get_state(scene.reference_state, next_step)
    actor_update = advanced[:, None] & scene.actor_mask
    next_state = select_state(actor_update, recorded_state, state)
    done = episode.done | (next_step >= scene.num_frames - 1)
    return StepResult(
        state=next_state,
        episode=EpisodeState(step=next_step, done=done),
        advanced=advanced,
        just_finished=done & ~episode.done,
        actor_mask=scene.reference_valid[
            torch.arange(next_step.shape[0], device=next_step.device), next_step
        ],
        static_mask=scene.static.valid[
            torch.arange(next_step.shape[0], device=next_step.device), next_step
        ],
    )
