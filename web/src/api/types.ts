export type ActorType = "vehicle" | "pedestrian" | "static";

export interface CaseSummary {
  id: string;
  scenario_id: string;
  source_kind: "demo" | "scene" | "replay";
  num_frames: number;
  dt: number;
  duration_s: number;
  actor_count: number;
  dynamic_actor_count?: number;
  static_object_count?: number;
}

export interface Actor {
  id: string;
  slot: number;
  type: ActorType;
  length: number;
  width: number;
  reference_to_center: number;
  is_ego?: boolean;
  blind?: boolean;
  control_type?: ControlKind | null;
  /** The fixed goal this rollout used instead of the route goal. */
  goal_override?: [number, number] | null;
  /** Set on actors this rollout added to the scene. */
  added_serial?: number | null;
  route?: [number, number][];
  route_source?: "none" | "map" | "gt";
  has_rail?: boolean;
  reactive_role?: ReactiveRole | null;
  route_start_s?: number;
  route_horizon_s?: number;
  static_position?: [number, number] | null;
  static_heading?: number;
  presence_intervals?: [number, number][];
}

export interface CaseDetail extends CaseSummary {
  actors: Actor[];
  bounds: [number, number, number, number];
  has_map: boolean;
}

export type MapPolylineKind =
  | "lane_centerline"
  | "lane_connector_centerline"
  | "lane_boundary"
  | "road_boundary"
  | "crosswalk"
  | "stop_line";

export interface SceneMap {
  format: "scene_map";
  schema_version: 1;
  scenario_id: string;
  map_name: string;
  coordinate_frame: "scene";
  world_from_scene: { x: number; y: number; heading: number };
  bounds: [number, number, number, number];
  polylines: {
    id: string;
    kind: MapPolylineKind;
    points: [number, number][];
  }[];
}

export interface ActorState {
  id: string;
  position: [number, number];
  center: [number, number];
  heading: number;
  velocity: [number, number];
  speed: number;
  acceleration: number;
  acceleration_command: number;
  steering: number;
  yaw_rate: number;
  goals?: [number, number][];
  log_pose?: { center: [number, number]; heading: number } | null;
}

export interface Frame {
  index: number;
  time_s: number;
  done: boolean;
  actors: ActorState[];
  /** Evaluator events of the ego at this frame. */
  ego_events?: ("collision" | "ttc" | "offroad")[];
}

export interface FrameChunk {
  case_id: string;
  start: number;
  total_frames: number;
  frames: Frame[];
}

export interface RolloutSource extends CaseSummary {
  num_actor_slots: number;
  has_map: boolean;
  /** The scene directory's name, e.g. val14. */
  collection?: string;
  scenario_type?: string | null;
}

export interface RolloutSettings {
  device: "auto" | "cpu" | "cuda";
  compiled: boolean;
  max_actors: number;
  max_static_objects?: number;
  max_frames: number;
  max_map_segments: number;
  max_batch_size: number;
}

export type LearnedKind = "single_frame";

export interface PolicyCheckpoint {
  name: string;
  path: string;
  /** Empty: SingleFrameNet checkpoints embed their config. */
  config_path: string;
  kind?: LearnedKind;
}

export interface RolloutSources {
  sources: RolloutSource[];
  settings: RolloutSettings;
  policy_checkpoints?: PolicyCheckpoint[];
}

export interface RolloutWorld {
  source_id: string;
  detail: CaseDetail;
  frames: Frame[];
  map: SceneMap | null;
}

export type PolicyKind = "log" | "rule_based" | LearnedKind;
export type ControlPreset =
  | "log"
  | "ego_idm"
  | "idm_pedestrian"
  | "ego_single_frame"
  | "ego_single_frame_reactive"
  | "single_frame_selfplay"
  | "nuplan_reactive";
export type AgentControlKind = "log" | "idm" | "pedestrian" | "absent" | LearnedKind;
/** What nuPlan's reactive (R) simulation does with an actor. */
export type ReactiveRole = "absent" | "idm" | "open_loop" | "ego";
export type ControlKind = AgentControlKind | "static";

export interface AgentControl {
  source_id: string;
  actor_id: string;
  control_type: AgentControlKind;
  blind?: boolean;
  /** A fixed goal (scene x, y in meters) replacing the learned policy's route goal. */
  goal?: [number, number];
}

export type AddedKind = "vehicle" | "pedestrian" | "static";

/** A stationary actor placed into one scene: box center, heading (rad), size (m). */
export interface AddedActor {
  source_id: string;
  /** Kept while the actor exists; the server derives its id from it. */
  serial: number;
  kind: AddedKind;
  position: [number, number];
  heading: number;
  length: number;
  width: number;
}

export type SamplingMethod = "argmax" | "expectation";

export interface RolloutOptions {
  checkpoint?: string;
  checkpoint_config?: string;
  preset?: ControlPreset;
  agent_controls?: AgentControl[];
  added_actors?: AddedActor[];
  sampling_method?: SamplingMethod;
  /** Summaries and scores only; fetch each world with getWorld. */
  lazy?: boolean;
  /** SingleFrameNet route goal, seconds ahead. */
  goal_horizon_s?: number;
}

/** The ego's nuPlan-style episode scores (training evaluator). */
export interface WorldMetrics {
  score: number;
  no_ego_at_fault_collisions: number;
  drivable_area_compliance: number;
  ego_is_making_progress: number;
  time_to_collision_within_bound: number;
  speed_limit_compliance: number;
  ego_progress_along_expert_route: number;
  ego_is_comfortable: number;
}

export interface WorldSummary {
  index: number;
  source_id: string;
  scenario_id: string;
  scenario_type?: string | null;
  collection?: string;
  num_frames: number;
  dynamic_actor_count: number;
  static_object_count: number;
  metrics?: WorldMetrics | null;
}

export interface BatchRollout {
  id: string;
  policy: PolicyKind;
  preset?: ControlPreset;
  pedestrian_oracle?: boolean;
  control_ego?: boolean;
  device: "cpu" | "cuda";
  compiled: boolean;
  batch_size: number;
  elapsed_s: number;
  world_transitions: number;
  worlds: RolloutWorld[];
  summaries?: WorldSummary[];
  /** Scenario ids skipped for exceeding the visualizer's capacities. */
  skipped?: string[];
}

/** What one SingleFrameNet actor kept at every frame; chunk geometry is listed once. */
export interface ObservationTrack {
  actor_id: string;
  agent_capacity: number;
  lane_capacity: number;
  blind: boolean;
  chunks: { points: [number, number][]; type: MapPolylineKind }[];
  /** Per frame: kept actor ids (the query first) and indices into `chunks`. */
  frames: { agents: string[]; lanes: number[] }[];
}

/** Agents and map chunks one SingleFrameNet actor kept at one frame. */
export interface ObservationView {
  actor_id: string;
  frame: number;
  /** Kept actor ids; the query itself comes first. */
  agents: string[];
  agent_capacity: number;
  /** Kept map chunks as ordered points in scene coordinates. */
  lanes: [number, number][][];
  lane_types: MapPolylineKind[];
  lane_capacity: number;
  blind: boolean;
}

/** One SingleFrameNet actor's decisions per frame (the action applied from that frame). */
export interface PolicyTrack {
  actor_id: string;
  jerk_range: [number, number];
  steering_rate_range: [number, number];
  gamma: number;
  /** Beta(alpha, beta) over unit [0, 1] for [jerk, steering rate]; null without a decision. */
  alpha: ([number, number] | null)[];
  beta: ([number, number] | null)[];
  /** Applied jerk [m/s^3] and steering rate [rad/s]. */
  action: ([number, number] | null)[];
  value: (number | null)[];
  /** Ego only: each reward term per step (reward, hard, goal, ttc, speed, ...). */
  rewards?: Record<string, (number | null)[]> | null;
}
