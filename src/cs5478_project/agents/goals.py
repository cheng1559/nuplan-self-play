"""Policy goals in scene coordinates: route goals for evaluation, log goals for training."""

import torch

from cs5478_project.agents.route import project_route, sample_route


def sample_goals(state, scene, routes, config):
    """Route goals [B, A, 1, 2], max(speed, goal_min_speed) * goal_horizon_s ahead along
    each actor's route (ego's map route, other actors' logged routes)."""
    progress = project_route(routes, state.position).progress
    lookahead = state.velocity.norm(dim=-1).clamp_min(config.goal_min_speed) * config.goal_horizon_s
    final_progress = torch.minimum(progress + lookahead, routes.total)
    return sample_route(routes, final_progress).unsqueeze(2)


GOAL_SNAP_KEEP = 0.5  # probability of keeping the logged position
GOAL_SNAP_LANES = 4  # nearest lane centerlines considered
GOAL_SNAP_DISTANCE = 20.0  # meters


def sample_training_goals(scene, goal_count: int = 1, generator=None):
    """Training goals [B, A, G, 2] from each actor's own log.

    Each actor gets 1 to G goals (equally likely): distinct future frames of its log
    drawn with probability proportional to t + 1, in time order; unused slots repeat
    the last goal. Each goal keeps its logged position with probability 1/2, else moves
    to its projection on one of the 4 nearest lane centerlines within 20 m (closer ones
    likelier). Actors without a valid future keep their first position.
    """
    from cs5478_project.datatypes.map import MapElementType

    valid = scene.reference_valid[:, 1:]  # [B, F - 1, A]
    b, steps, a = valid.shape
    device = valid.device
    frames = torch.arange(1, steps + 1, device=device)
    inside = valid & (frames[None, :, None] < scene.num_frames[:, None, None])
    weights = ((frames[None, :, None] + 1).float() * inside).permute(0, 2, 1)  # [B, A, F - 1]
    has_future = weights.sum(-1) > 0
    weights = torch.where(has_future[..., None], weights, 1.0)
    # Distinct frames without replacement (Gumbel top-k), then the goal count.
    uniform = torch.rand(weights.shape, device=device, generator=generator).clamp(1e-9, 1 - 1e-9)
    score = torch.where(weights > 0, weights.log() - (-uniform.log()).log(), -torch.inf)
    picked = score.topk(min(goal_count, steps), -1).indices + 1  # [B, A, k]
    count = torch.randint(1, goal_count + 1, (b, a), device=device, generator=generator)
    count = torch.minimum(count, inside.sum(1).clamp_min(1))
    slot = torch.arange(picked.shape[-1], device=device)
    picked = torch.where(slot < count[..., None], picked, steps + 1).sort(-1).values
    last = picked.gather(-1, (count - 1)[..., None])
    picked = torch.where(slot < count[..., None], picked, last)
    picked = torch.where(has_future[..., None], picked, 0)
    if picked.shape[-1] < goal_count:
        picked = torch.cat(
            (picked, picked[..., -1:].expand(-1, -1, goal_count - picked.shape[-1])), -1
        )
    log = scene.reference_state.position  # [B, F, A, 2]
    world = torch.arange(b, device=device)[:, None, None]
    actor = torch.arange(a, device=device)[None, :, None]
    goals = log[world, picked, actor]  # [B, A, G, 2]

    # Snap half of them to a nearby lane centerline.
    kind = scene.map.chunk_type
    lane = scene.map.chunk_mask & (
        (kind == MapElementType.LANE_CENTERLINE)
        | (kind == MapElementType.LANE_CONNECTOR_CENTERLINE)
    )  # [B, C]
    points = scene.map.chunk_points  # [B, C, P, 2]
    flat = goals.view(b, a * goal_count, 2)
    middle = points[:, :, points.shape[2] // 2]
    rough = (flat[:, :, None] - middle[:, None]).norm(dim=-1).masked_fill(~lane[:, None], torch.inf)
    near = rough.topk(min(16, rough.shape[-1]), -1, largest=False).indices  # [B, N, 16]
    chunk = points[torch.arange(b, device=device)[:, None, None], near]  # [B, N, 16, P, 2]
    start, delta = chunk[..., :-1, :], chunk[..., 1:, :] - chunk[..., :-1, :]
    p = flat[:, :, None, None]
    t = (((p - start) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-9)).clamp(0, 1)
    foot = start + t[..., None] * delta
    gap = (p - foot).norm(dim=-1)  # [B, N, 16, P - 1]
    gap, segment = gap.min(-1)
    gap = gap.masked_fill(~lane.gather(1, near.flatten(1)).view_as(gap), torch.inf)
    foot = foot.gather(3, segment[..., None, None].expand(-1, -1, -1, 1, 2))[..., 0, :]
    gap, order = gap.topk(min(GOAL_SNAP_LANES, gap.shape[-1]), -1, largest=False)
    foot = foot.gather(2, order[..., None].expand(-1, -1, -1, 2))
    usable = gap <= GOAL_SNAP_DISTANCE
    weight = torch.where(usable, 1 / gap.clamp_min(1e-3), 0.0)
    choice = torch.multinomial(
        torch.where(usable.any(-1, keepdim=True), weight, 1.0).flatten(0, 1), 1,
        generator=generator,
    ).view(b, -1)  # fmt: skip
    snapped = foot.gather(2, choice[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
    snap = usable.any(-1) & (
        torch.rand(usable.shape[:2], device=device, generator=generator) >= GOAL_SNAP_KEEP
    )
    goals = torch.where(snap[..., None], snapped, flat).view(b, a, goal_count, 2)
    # Unused slots repeat the last real goal (after snapping).
    used = torch.arange(goal_count, device=device) < count[..., None]
    last = goals.gather(2, (count - 1)[..., None, None].expand(-1, -1, 1, 2))
    return torch.where(used[..., None], goals, last)
