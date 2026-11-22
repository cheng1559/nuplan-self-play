"""Per-frame training rewards and episode nuPlan-style metrics for chosen actors.

Every evaluated actor is scored the way nuPlan scores ego, against all other actors and
static objects. The reward reuses the metric's
per-frame geometry:

    reward = hard + (1 - done) * (goal + ttc * speed * comfort * centerline * direction / steps)

``hard`` is ``collision_reward`` on any overlap while the actor moves, a speed-scaled
``offroad_reward`` when the box touches a road boundary, and ``wrong_way_reward`` on
severe wrong-way driving; each ends the actor's episode (``done``). ``goal`` is paid once
when the actor passes within ``goal_threshold`` of its goal. The soft scores are each in
[0, 1]. The metrics keep nuPlan's rules (at-fault collisions, 0.3 m off-road, and so on).
"""

from typing import NamedTuple

import torch
from torch import Tensor

from cs5478_project.agents.route import cache_routes, project_route
from cs5478_project.datatypes.map import MapElementType
from cs5478_project.dynamics.bicycle import wrap_angle
from cs5478_project.evaluation.config import EvaluationConfig
from cs5478_project.evaluation.frame import COLLISION_CLASSES, evaluate_frame
from cs5478_project.evaluation.geometry import boundary_crossing
from cs5478_project.evaluation.savgol import savgol

# nuPlan's comfort filters: 8-sample acceleration smoothing, then 15-sample jerk, and a
# 5-sample window for yaw rate and yaw acceleration.
ACCEL_WINDOW, JERK_WINDOW, YAW_WINDOW = 8, 15, 5
# Allowed at-fault collisions per class before the collision score reaches zero.
COLLISION_ALLOWANCE = (0, 0, 1)


class Reward(NamedTuple):
    """[B, A] per-step reward and its parts; zero for actors not evaluated or finished."""

    reward: Tensor
    done: Tensor  # this step ended the actor's episode
    terminated: Tensor  # the actor's episode has ended
    hard: Tensor
    goal: Tensor
    ttc: Tensor
    speed: Tensor
    comfort: Tensor
    centerline: Tensor
    direction: Tensor
    pace: Tensor  # the soft product's weight this step (route-completion gain or 1/steps)


class Metrics(NamedTuple):
    """nuPlan-style episode scores per evaluated actor, [Q] each."""

    world: Tensor
    slot: Tensor
    no_ego_at_fault_collisions: Tensor
    drivable_area_compliance: Tensor
    ego_is_making_progress: Tensor
    time_to_collision_within_bound: Tensor
    speed_limit_compliance: Tensor
    ego_progress_along_expert_route: Tensor
    ego_is_comfortable: Tensor
    score: Tensor


def comfort_signals(series: Tensor, length: Tensor, dt: Tensor) -> tuple[Tensor, ...]:
    """nuPlan's comfort signals from [N, T, 3] (center lon accel, |accel|, heading)."""
    accel = savgol(series[..., 0], length, ACCEL_WINDOW, 2)
    magnitude = savgol(series[..., 1], length, ACCEL_WINDOW, 2)
    dt = dt[:, None]
    jerk = savgol(accel, length, JERK_WINDOW, 2, 1) / dt
    jerk_magnitude = savgol(magnitude, length, JERK_WINDOW, 2, 1) / dt
    yaw_rate = savgol(series[..., 2], length, YAW_WINDOW, 2, 1) / dt
    yaw_accel = savgol(series[..., 2], length, YAW_WINDOW, 3, 2) / dt.square()
    return accel, jerk, jerk_magnitude, yaw_rate, yaw_accel


def within_comfort(signals, config: EvaluationConfig) -> Tensor:
    accel, jerk, jerk_magnitude, yaw_rate, yaw_accel = signals
    return (
        (accel >= config.min_lon_accel)
        & (accel <= config.max_lon_accel)
        & (jerk.abs() <= config.max_abs_lon_jerk)
        & (jerk_magnitude.abs() <= config.max_abs_mag_jerk)
        & (yaw_rate.abs() <= config.max_abs_yaw_rate)
        & (yaw_accel.abs() <= config.max_abs_yaw_accel)
    )


class DrivingEvaluator:
    """Score a fixed set of actors from reset to the end of the episode.

    Call ``reset`` after the environment resets (it also checks the first frame), then
    ``step`` after every environment step. Only full resets are supported.
    """

    def __init__(self, config: EvaluationConfig = EvaluationConfig(), *, compile_step=False):
        self.config = config
        self._frame = (
            # dynamic=None: the evaluated-actor count varies by batch in self-play.
            torch.compile(evaluate_frame, fullgraph=True, dynamic=None)
            if compile_step
            else evaluate_frame
        )

    @torch.no_grad()
    def reset(
        self,
        scene,
        state,
        step,
        actor_mask,
        static_mask,
        control_types,
        evaluated,
        goals=None,
        weights=None,
    ):
        """``evaluated`` [B, A] selects the actors to score; it is fixed for the episode.

        ``goals`` [B, A, 2] are the goal-reward targets (the policy's fixed goals in
        training); by default each actor's last logged position in the episode.
        ``weights`` is a RewardWeights of [B, A] tensors (domain randomization); by
        default the config's reward, which every weight reproduces at its default.
        """
        self.scene = scene
        self.query_world, self.query_slot = evaluated.nonzero(as_tuple=True)
        qw, qs = self.query_world, self.query_slot
        q, device = qw.shape[0], qw.device
        frames = scene.reference_state.position.shape[1]
        objects = state.position.shape[1] + scene.static.position.shape[1]
        self._shape = evaluated.shape
        self._dt = scene.dt[qw]
        self._steps = (scene.num_frames[qw] - 1).clamp_min(1).to(self._dt.dtype)

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros((q, *shape), dtype=dtype, device=device)

        self._collided = zeros(objects, dtype=torch.bool)
        self._at_fault = zeros(COLLISION_CLASSES, dtype=torch.int64)
        self._ttc_violation = zeros(dtype=torch.bool)
        self._drivable_violation = zeros(dtype=torch.bool)
        self._overspeed = zeros()
        self._frames = zeros(dtype=torch.int64)
        self._first = torch.full((q,), -1, dtype=torch.int64, device=device)
        self._series = zeros(frames, 3)
        # The comfort reward refilters the whole history each step; it needs this many frames.
        self._history = min(frames, int(scene.num_frames.max()))
        # Per-step progress along the nearest centerline over the driving-direction horizon.
        horizon = max(1, round(self.config.wrong_way_horizon / float(self._dt.max())))
        self._along = zeros(horizon)
        self._ttc_triggered = zeros(dtype=torch.bool)
        self._comfort_triggered = zeros(dtype=torch.bool)
        self._goal_reached = zeros(dtype=torch.bool)
        self._terminated = zeros(dtype=torch.bool)
        self._yaw_rate = state.yaw_rate[qw, qs].clone()
        self._heading = state.heading[qw, qs].clone()
        self._position = state.position[qw, qs].clone()

        # Expert (log) start and end within the episode, for progress and the goal.
        self._routes = cache_routes(scene.routes)
        valid = scene.reference_valid  # [B, F, A]
        index = torch.arange(frames, device=device)[None, :, None]
        inside = valid & (index < scene.num_frames[:, None, None])
        first = torch.where(inside, index, frames).amin(1).clamp_max(frames - 1)  # [B, A]
        last = torch.where(inside, index, -1).amax(1).clamp_min(0)
        log = scene.reference_state.position  # [B, F, A, 2]

        def at(frame):
            return log.gather(1, frame[:, None, :, None].expand(-1, 1, -1, 2))[:, 0]

        start, end = at(first), at(last)
        expert = (
            project_route(self._routes, end).progress - project_route(self._routes, start).progress
        )
        self._expert_progress = expert[qw, qs]
        # [Q, G, 2]: goals [B, A, G, 2] (or [B, A, 2]); by default the log's end.
        goal = end[:, :, None] if goals is None else goals
        self._goal = (goal if goal.ndim == 4 else goal[:, :, None])[qw, qs]
        self._weights = None if weights is None else type(weights)(*(w[qw, qs] for w in weights))
        self._start_progress = project_route(self._routes, state.position).progress[qw, qs]
        self._progress = self._start_progress.clone()
        frame, _ = self._observe(
            state, step, actor_mask, static_mask, control_types, first_frame=True
        )
        self._centerline_chunk = frame.centerline_chunk
        self._box = frame.box

    def _observe(self, state, step, actor_mask, static_mask, control_types, first_frame=False):
        qw, qs, config = self.query_world, self.query_slot, self.config
        frame = self._frame(
            state, self.scene, step, actor_mask, static_mask, control_types, qw, qs,
            self._collided, config,
        )  # fmt: skip
        present = frame.present
        self._collided.scatter_(
            1, frame.track, frame.new_collision | self._collided.gather(1, frame.track)
        )
        self._at_fault += frame.at_fault_by_class
        self._ttc_violation |= present & (frame.ttc <= config.least_min_ttc)
        self._drivable_violation |= present & (frame.outside > config.drivable_violation)
        self._overspeed += torch.where(present, frame.overspeed * self._dt, 0.0)
        self._frames += present.long()
        self._first = torch.where(present & (self._first < 0), step[qw], self._first)

        # nuPlan's center acceleration: rear-axle acceleration plus d * (w^2 + dw/dt).
        yaw_rate = state.yaw_rate[qw, qs]
        yaw_accel = (
            torch.zeros_like(yaw_rate) if first_frame else (yaw_rate - self._yaw_rate) / self._dt
        )
        offset = self.scene.reference_to_center[qw, qs]
        center_accel = state.acceleration[qw, qs] + offset * (yaw_rate.square() + yaw_accel)
        heading = self._heading + wrap_angle(state.heading[qw, qs] - self._heading)
        values = torch.stack((center_accel, state.acceleration[qw, qs].abs(), heading), -1)
        rows = torch.arange(qw.shape[0], device=qw.device)
        frame_index = step[qw].clamp_max(self._series.shape[1] - 1)
        self._series[rows, frame_index] = torch.where(
            present[:, None], values, self._series[rows, frame_index]
        )
        self._yaw_rate = torch.where(present, yaw_rate, self._yaw_rate)
        self._heading = torch.where(present, heading, self._heading)
        progress = project_route(self._routes, state.position).progress[qw, qs]
        self._progress = torch.where(present, progress, self._progress)
        return frame, frame_index

    def _comfortable_now(self, frame_index):
        """Per-frame comfort: acceleration and yaw bounds at the latest frame, and
        jerk bounds on the largest jerk anywhere in the history, refiltered up to now (so a
        jerk violation fails every later frame)."""
        config, history = self.config, self._history
        window = torch.arange(history, device=frame_index.device) - (history - 1)
        first = self._first.clamp_min(0)[:, None]
        index = frame_index[:, None] + window
        rows = torch.arange(index.shape[0], device=index.device)[:, None]
        series = self._series[rows, index.clamp_min(first)]  # earlier frames repeat the first
        length = torch.full_like(frame_index, history)
        accel, jerk, jerk_magnitude, yaw_rate, yaw_accel = comfort_signals(series, length, self._dt)
        observed = index >= first
        jerk = torch.where(observed, jerk.abs(), 0.0).amax(-1)
        jerk_magnitude = torch.where(observed, jerk_magnitude.abs(), 0.0).amax(-1)
        latest = (accel[:, -1], jerk, jerk_magnitude, yaw_rate[:, -1], yaw_accel[:, -1])
        return within_comfort(latest, config)

    @torch.no_grad()
    def step(self, state, step, actor_mask, static_mask, control_types, advanced) -> Reward:
        """Score the transition into ``state`` at ``step`` for worlds that ``advanced``."""
        qw, qs, config = self.query_world, self.query_slot, self.config
        position = state.position[qw, qs]
        frame, frame_index = self._observe(state, step, actor_mask, static_mask, control_types)
        active = frame.present & advanced[qw] & ~self._terminated

        collided = frame.moving_collision
        # Off-road check: the box, or its corners' paths, touches a road boundary.
        offroad = frame.present & boundary_crossing(
            self.scene.map, qw, frame.box, self._box, MapElementType.ROAD_BOUNDARY
        )
        # This step's events per evaluated actor [Q] (the visualizer colors them): an
        # overlap or at-fault collision, a metric TTC violation, leaving the drivable area.
        self.events = {
            "collision": frame.present & (collided | (frame.at_fault_by_class.sum(-1) > 0)),
            "ttc": frame.present & (frame.ttc <= config.least_min_ttc),
            "offroad": frame.present & (offroad | (frame.outside > config.drivable_violation)),
        }
        self._box = torch.where(frame.present[:, None, None], frame.box, self._box)
        speed_now = state.velocity[qw, qs].norm(dim=-1)

        # Driving direction: progress along the nearest centerline over the last second,
        # counted while that centerline chunk stays the same.
        along = ((position - self._position) * frame.centerline_direction).sum(-1)
        near_centerline = frame.centerline_offset <= config.centerline_band
        same_line = frame.centerline_chunk == self._centerline_chunk
        along = torch.where(frame.present & near_centerline & same_line, along, 0.0)
        self._centerline_chunk = torch.where(
            frame.present, frame.centerline_chunk, self._centerline_chunk
        )
        self._along = torch.cat((self._along[:, 1:], along[:, None]), 1)
        backward = -self._along.sum(-1)
        wrong_way = (backward > config.wrong_way_threshold) & ~offroad
        severe = (backward > config.wrong_way_severe) & ~offroad

        offroad_scale = 1 + config.offroad_speed_gain * torch.tanh(
            2 * speed_now / config.offroad_speed_norm
        )
        weights = self._weights
        collision_reward = config.collision_reward if weights is None else weights.collision
        offroad_reward = config.offroad_reward if weights is None else weights.offroad
        hard = collided * collision_reward + torch.where(
            offroad,
            offroad_reward * offroad_scale,
            severe * config.wrong_way_reward,
        )
        done = active & (collided | offroad | severe)

        # The goal counts once, when the step's swept segment passes near it.
        # Reaching any of the goals pays the bonus once.
        a, delta = self._position[:, None], (position - self._position)[:, None]
        fraction = ((self._goal - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)
        nearest = a + fraction.clamp(0, 1)[..., None] * delta
        threshold = config.goal_threshold if weights is None else weights.goal_threshold[:, None]
        reached = (
            active & ~self._goal_reached & ((self._goal - nearest).norm(dim=-1) < threshold).any(-1)
        )
        self._goal_reached |= reached
        self._position = torch.where(frame.present[:, None], position, self._position)

        ttc = (
            (frame.ttc_wide - config.least_min_ttc) / (config.ttc_horizon - config.least_min_ttc)
        ).clamp(0, 1)
        ttc = ttc * torch.where(self._ttc_triggered, config.ttc_repeat_factor, 1.0)
        self._ttc_triggered |= active & (frame.ttc_wide <= config.least_min_ttc)
        speed = (1 - frame.overspeed / config.overspeed_tolerance).clamp(0, 1)
        comfortable = self._comfortable_now(frame_index)
        comfort = torch.where(comfortable, 1.0, config.comfort_violation_score)
        comfort = comfort * torch.where(self._comfort_triggered, config.comfort_repeat_factor, 1.0)
        self._comfort_triggered |= active & ~comfortable

        # Centerline: the share of lateral room used before the box presses a lane line.
        offset, boundary = frame.centerline_offset, frame.boundary_offset
        half_width = self.scene.size[qw, qs, 1] * 0.5
        used = offset / (offset + boundary - half_width).clamp_min(1e-6)
        used = torch.where(boundary <= half_width, 1.0, used)
        used = torch.where(used.isfinite(), used, 1.0).clamp(0, 1)
        centerline = 1 - (1 - config.centerline_score) * used
        centerline = torch.where((boundary < config.centerline_band) & ~wrong_way, centerline, 1.0)
        direction = torch.where(wrong_way, config.wrong_way_score, 1.0)

        goal = reached * (config.goal_reward if weights is None else weights.goal)
        if weights is not None:
            # Penalty strengths: 1 is the default reward, 0 ignores the term.
            ttc = 1 - weights.ttc * (1 - ttc)
            comfort = 1 - weights.comfort * (1 - comfort)
            speed = 1 - weights.speed * (1 - speed)
        soft = ttc * speed * comfort * centerline * direction
        pace = 1 / self._steps
        reward = hard + (~done) * (goal + soft * pace)
        self._terminated |= done

        def scatter(value):
            output = torch.zeros(self._shape, dtype=value.dtype, device=value.device)
            output[qw, qs] = torch.where(active, value, torch.zeros_like(value))
            return output

        terminated = torch.zeros(self._shape, dtype=torch.bool, device=qw.device)
        terminated[qw, qs] = self._terminated
        return Reward(
            scatter(reward),
            scatter(done),
            terminated,
            scatter(hard),
            scatter(goal),
            scatter(ttc),
            scatter(speed),
            scatter(comfort),
            scatter(centerline),
            scatter(direction),
            scatter(pace * torch.ones_like(soft)),
        )

    @property
    def goal_reached(self) -> Tensor:
        """[Q] whether each evaluated actor has already passed within reach of its goal."""
        return self._goal_reached

    @torch.no_grad()
    def metrics(self) -> Metrics:
        config = self.config
        allowance = torch.tensor(COLLISION_ALLOWANCE, device=self._at_fault.device)
        collisions = (1 - self._at_fault / (allowance + 1)).clamp_min(0).prod(-1)
        drivable = (~self._drivable_violation).float()
        ttc = (~self._ttc_violation).float()
        duration = (self._frames - 1).clamp_min(0) * self._dt
        speed = torch.where(
            duration > 0,
            (1 - self._overspeed / (config.max_overspeed * duration.clamp_min(1e-9))).clamp_min(0),
            1.0,
        )
        progress = self._progress - self._start_progress
        threshold = config.progress_threshold
        ratio = (
            progress.clamp_min(threshold) / self._expert_progress.clamp_min(threshold)
        ).clamp_max(1)
        ratio = torch.where(progress < -threshold, 0.0, ratio)
        making_progress = (ratio >= config.min_progress_ratio).float()

        # Comfort over the whole episode, from each actor's first frame.
        steps = self._series.shape[1]
        index = self._first.clamp_min(0)[:, None] + torch.arange(steps, device=self._first.device)
        rows = torch.arange(index.shape[0], device=index.device)[:, None]
        series = self._series[rows, index.clamp_max(steps - 1)]
        length = self._frames
        signals = comfort_signals(series, length, self._dt)
        valid = torch.arange(steps, device=index.device)[None] < length[:, None]
        comfortable = (within_comfort(signals, config) | ~valid).all(-1)
        comfortable = (comfortable | (length < JERK_WINDOW)).float()

        weighted = (
            config.weight_ttc * ttc
            + config.weight_progress * ratio
            + config.weight_speed * speed
            + config.weight_comfort * comfortable
        ) / (
            config.weight_ttc + config.weight_progress + config.weight_speed + config.weight_comfort
        )
        score = collisions * drivable * making_progress * weighted
        return Metrics(
            self.query_world,
            self.query_slot,
            collisions,
            drivable,
            making_progress,
            ttc,
            speed,
            ratio,
            comfortable,
            score,
        )
