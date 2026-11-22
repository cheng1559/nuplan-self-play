"""Typed contracts for live batch simulation and archived replay inspection."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cs5478_project.defaults import EPISODE_FRAMES, MAX_DYNAMIC_ACTORS, MAX_STATIC_OBJECTS
from cs5478_project.visualization.map import SceneMap

SourceKind = Literal["demo", "scene", "replay"]
ActorKind = Literal["vehicle", "pedestrian", "static"]
PolicyKind = Literal["log", "rule_based", "single_frame"]
ControlPreset = Literal[
    "log",
    "ego_idm",
    "idm_pedestrian",
    "ego_single_frame",
    "ego_single_frame_reactive",
    "single_frame_selfplay",
    "nuplan_reactive",
]
AgentControlKind = Literal["log", "idm", "pedestrian", "single_frame", "absent"]
ReactiveRoleKind = Literal["absent", "idm", "open_loop", "ego"]


class CaseSummary(BaseModel):
    id: str
    scenario_id: str
    source_kind: SourceKind
    num_frames: int
    dt: float
    duration_s: float
    actor_count: int
    dynamic_actor_count: int = 0
    static_object_count: int = 0


class CaseList(BaseModel):
    cases: list[CaseSummary]


class Actor(BaseModel):
    id: str
    slot: int
    type: ActorKind
    length: float
    width: float
    reference_to_center: float
    is_ego: bool = False
    blind: bool = False
    control_type: AgentControlKind | Literal["static"] | None = None
    # The fixed goal this rollout used instead of the route goal, if one was set.
    goal_override: tuple[float, float] | None = None
    # Set on actors this rollout added to the scene: the request's serial.
    added_serial: int | None = None
    route_start_s: float = 0.0
    route_horizon_s: float = 0.0
    route: list[tuple[float, float]] = Field(default_factory=list)
    route_source: Literal["none", "map", "gt"] = "none"
    # IDM control needs an exported rail; the role is what nuPlan R does with the actor.
    has_rail: bool = False
    reactive_role: ReactiveRoleKind | None = None
    static_position: tuple[float, float] | None = None
    static_heading: float = 0.0
    presence_intervals: list[tuple[int, int]] = Field(default_factory=list)


class CaseDetail(CaseSummary):
    actors: list[Actor]
    bounds: tuple[float, float, float, float]
    has_map: bool = False


class LogPose(BaseModel):
    center: tuple[float, float]
    heading: float


class ActorState(BaseModel):
    id: str
    position: tuple[float, float]
    center: tuple[float, float]
    heading: float
    velocity: tuple[float, float]
    speed: float
    acceleration: float
    acceleration_command: float
    steering: float
    yaw_rate: float
    goals: list[tuple[float, float]] = Field(default_factory=list)
    log_pose: LogPose | None = None


class Frame(BaseModel):
    index: int
    time_s: float
    done: bool
    actors: list[ActorState]
    # Evaluator events of the ego at this frame: "collision", "ttc", "offroad".
    ego_events: list[str] = Field(default_factory=list)


class FrameChunk(BaseModel):
    case_id: str
    start: int
    total_frames: int
    frames: list[Frame]


class RolloutSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    device: Literal["auto", "cpu", "cuda"] = "auto"
    compiled: bool = False
    max_actors: Annotated[int, Field(strict=True, ge=1)] = MAX_DYNAMIC_ACTORS
    max_static_objects: Annotated[int, Field(strict=True, ge=1)] = MAX_STATIC_OBJECTS
    max_frames: Annotated[int, Field(strict=True, ge=1)] = EPISODE_FRAMES
    max_map_segments: Annotated[int, Field(strict=True, ge=1)] = 2048
    max_batch_size: Annotated[int, Field(strict=True, ge=1)] = 256


class RolloutSource(CaseSummary):
    num_actor_slots: int
    has_map: bool
    # The artifact root's directory name (e.g. val14) and nuPlan's scenario type.
    collection: str = ""
    scenario_type: str | None = None


class PolicyCheckpoint(BaseModel):
    name: str
    path: str
    # Empty: SingleFrameNet checkpoints embed their config.
    config_path: str = ""
    kind: Literal["single_frame"] = "single_frame"


class RolloutSourceList(BaseModel):
    sources: list[RolloutSource]
    settings: RolloutSettings
    policy_checkpoints: list[PolicyCheckpoint] = Field(default_factory=list)


class AgentControl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: Annotated[str, Field(strict=True, min_length=1)]
    actor_id: Annotated[str, Field(strict=True, min_length=1)]
    control_type: AgentControlKind
    blind: bool = False
    # A fixed goal (scene x, y in meters) that replaces the learned policy's route goal.
    goal: tuple[float, float] | None = None


class AddedActor(BaseModel):
    """A stationary actor placed into one scene for this rollout."""

    model_config = ConfigDict(extra="forbid")

    source_id: Annotated[str, Field(strict=True, min_length=1)]
    # Chosen by the client and kept while the actor exists; its id derives from it.
    serial: Annotated[int, Field(strict=True, ge=0, lt=10_000)]
    kind: Literal["vehicle", "pedestrian", "static"]
    # Box center (scene x, y in meters), heading in radians, box size in meters.
    position: tuple[float, float]
    heading: float = 0.0
    length: Annotated[float, Field(gt=0, le=30)]
    width: Annotated[float, Field(gt=0, le=30)]


class BatchRolloutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: Annotated[
        list[Annotated[str, Field(strict=True, min_length=1)]], Field(min_length=1)
    ]
    preset: ControlPreset = "log"
    agent_controls: list[AgentControl] = Field(default_factory=list)
    added_actors: list[AddedActor] = Field(default_factory=list)
    checkpoint: str | None = None
    checkpoint_config: str | None = None
    sampling_method: Literal["argmax", "expectation"] = "argmax"
    # Return only per-world summaries and scores; fetch each world's frames on demand
    # (GET /rollouts/{id}/worlds/{index}). Oversized scenes are skipped, not rejected.
    lazy: bool = False
    # SingleFrameNet's route goal, seconds ahead (the checkpoint's own setting if unset).
    goal_horizon_s: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source_ids must not contain duplicates")
        keys = [(control.source_id, control.actor_id) for control in self.agent_controls]
        if len(keys) != len(set(keys)):
            raise ValueError("agent_controls must not contain duplicate source/actor pairs")
        if any(control.source_id not in self.source_ids for control in self.agent_controls):
            raise ValueError("agent_controls must reference selected sources")
        added = [(item.source_id, item.serial) for item in self.added_actors]
        if len(added) != len(set(added)):
            raise ValueError("added_actors must not repeat a serial within a source")
        if any(item.source_id not in self.source_ids for item in self.added_actors):
            raise ValueError("added_actors must reference selected sources")
        return self


class ObservedChunk(BaseModel):
    points: list[tuple[float, float]]  # scene-frame chunk points, in order
    type: str


class ObservedFrame(BaseModel):
    agents: list[str]  # kept actor ids, the query itself first
    lanes: list[int]  # indices into ObservationTrack.chunks


class ObservationTrack(BaseModel):
    """What one SingleFrameNet actor kept at every frame; chunk geometry is sent once."""

    actor_id: str
    agent_capacity: int
    lane_capacity: int
    blind: bool
    chunks: list[ObservedChunk]
    frames: list[ObservedFrame]


class RolloutWorld(BaseModel):
    source_id: str
    detail: CaseDetail
    frames: list[Frame]
    map: SceneMap | None


class PolicyTrack(BaseModel):
    """One SingleFrameNet actor's decisions per frame (the action applied from that frame).

    Beta(alpha, beta) per action dimension over unit [0, 1]: jerk spans jerk_range and
    steering rate steering_rate_range. ``rewards`` (ego only, when scored) holds each
    reward term per step; None marks frames without a value.
    """

    actor_id: str
    jerk_range: tuple[float, float]
    steering_rate_range: tuple[float, float]
    gamma: float
    alpha: list[tuple[float, float] | None]
    beta: list[tuple[float, float] | None]
    action: list[tuple[float, float] | None]  # applied jerk [m/s^3], steering rate [rad/s]
    value: list[float | None]
    rewards: dict[str, list[float | None]] | None = None


class WorldMetrics(BaseModel):
    """The ego's nuPlan-style episode scores, as the training evaluator computes them."""

    score: float
    no_ego_at_fault_collisions: float
    drivable_area_compliance: float
    ego_is_making_progress: float
    time_to_collision_within_bound: float
    speed_limit_compliance: float
    ego_progress_along_expert_route: float
    ego_is_comfortable: float


class WorldSummary(BaseModel):
    index: int
    source_id: str
    scenario_id: str
    scenario_type: str | None = None
    collection: str = ""
    num_frames: int
    dynamic_actor_count: int
    static_object_count: int
    metrics: WorldMetrics | None = None


class BatchRollout(BaseModel):
    id: str
    preset: ControlPreset = "log"
    policy: PolicyKind = "log"
    pedestrian_oracle: bool = False
    control_ego: bool = True
    device: Literal["cpu", "cuda"]
    compiled: bool
    batch_size: int
    elapsed_s: float
    world_transitions: int
    worlds: list[RolloutWorld]
    checkpoint: str | None = None
    summaries: list[WorldSummary] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # oversized scenario ids (lazy only)
