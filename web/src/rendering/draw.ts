import type {
  Actor,
  ActorState,
  CaseDetail,
  Frame,
  MapPolylineKind,
  ObservationView,
  SceneMap,
} from "@/api/types";
import { pointInBox, screenToWorld, worldToScreen } from "./camera";
import type { Camera, Point, Viewport } from "./camera";
import { layoutActorLabels, screenBoxBounds } from "./labels";
import type { ActorLabel } from "./labels";
import { frameActors } from "@/api/actors";

export const ACTOR_COLORS = {
  vehicle: "#4a90e2",
  pedestrian: "#b58336",
  static: "#8b929b",
} as const;

/** Ego colors for evaluator events, most severe first. */
export const EVENT_COLORS = {
  collision: "#dc2626",
  offroad: "#db2777",
  ttc: "#f59e0b",
} as const;
/** nuPlan's TTC bound and projection horizon [s] (EvaluationConfig). */
const TTC_BOUND = 0.95;
const TTC_HORIZON = 3.0;

function eventColor(frame: Frame | null): string | null {
  const events = frame?.ego_events ?? [];
  for (const name of ["collision", "offroad", "ttc"] as const)
    if (events.includes(name)) return EVENT_COLORS[name];
  return null;
}

/** The ego box swept forward at constant velocity: the TTC bound filled, the horizon outlined. */
function drawTtcRange(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  actor: Actor,
  state: ActorState,
  color: string,
): void {
  const center = worldToScreen(state.center, camera, viewport);
  const length = actor.length * camera.scale;
  const width = actor.width * camera.scale;
  const reach = (seconds: number) => state.speed * seconds * camera.scale;
  context.save();
  context.translate(center[0], center[1]);
  context.rotate(-state.heading);
  context.fillStyle = color;
  context.globalAlpha = 0.14;
  context.fillRect(-length / 2, -width / 2, length + reach(TTC_BOUND), width);
  context.globalAlpha = 0.7;
  context.strokeStyle = color;
  context.lineWidth = 1;
  context.setLineDash([5, 4]);
  context.strokeRect(-length / 2, -width / 2, length + reach(TTC_BOUND), width);
  context.globalAlpha = 0.4;
  context.setLineDash([2, 4]);
  context.strokeRect(-length / 2, -width / 2, length + reach(TTC_HORIZON), width);
  context.restore();
}

/**
 * Presentation style, opt-in with ?demo in the page URL (used to record slide videos): other
 * actors are grey, the ego takes the colour given by ?ego=RRGGBB, logged reference poses are
 * hidden, and only the ego's trail stands out.
 */
const PAGE = typeof window === "undefined" ? null : new URLSearchParams(window.location.search);
const DEMO = PAGE?.has("demo") ?? false;
const DEMO_EGO = `#${PAGE?.get("ego") ?? "2563eb"}`;
const DEMO_OTHER = "#9ca3af";
const DEMO_PEER = `#${PAGE?.get("peer") ?? "f2b27a"}`;

export function actorColor(actor: Actor): string {
  if (DEMO) {
    if (actor.is_ego) return DEMO_EGO;
    // other policy-driven agents (self-play) in a lighter shade of the ego colour
    if (actor.control_type && actor.control_type === "single_frame")
      return DEMO_PEER;
    return DEMO_OTHER;
  }
  if (actor.blind) return "#dc2626";
  if (actor.is_ego || (actor.control_type && !["log", "static"].includes(actor.control_type)))
    return "#2563eb";
  return ACTOR_COLORS[actor.type];
}

const CANVAS_FONT = '400 11px "Inter", Arial, sans-serif';

export interface DrawOptions {
  caseDetail: CaseDetail;
  frame: Frame | null;
  history: Frame[];
  selectedActorId: string | null;
  showTrails: boolean;
  showLabels: boolean;
  showMap: boolean;
  showRoute?: boolean;
  showGoals: boolean;
  /** The ego's TTC range (constant-velocity sweep). */
  showTtc?: boolean;
  map: SceneMap | null;
  /** The selected SingleFrameNet actor's kept agents and lanes, if shown. */
  observation?: ObservationView | null;
  /** The selected actor's fixed goal (scene coordinates), applied or still pending. */
  pinnedGoal?: Point | null;
  /** Added actors the shown rollout has not run with yet, drawn as dashed outlines. */
  pendingActors?: { position: Point; heading: number; length: number; width: number }[];
}

export const OBSERVATION_COLOR = "#7c3aed";
const GOAL_COLOR = "#ea580c";
const GOAL_RADIUS = 7;

function drawObservedLanes(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  observation: ObservationView,
): void {
  context.save();
  context.strokeStyle = OBSERVATION_COLOR;
  context.globalAlpha = 0.75;
  context.lineWidth = 2.5;
  context.lineCap = "round";
  context.beginPath();
  for (const points of observation.lanes) {
    points.forEach((point, index) => {
      const [x, y] = worldToScreen(point, camera, viewport);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
  }
  context.stroke();
  context.restore();
}

function drawObservedAgent(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  actor: Actor,
  state: ActorState,
): void {
  const center = worldToScreen(state.center, camera, viewport);
  const length = actor.length * camera.scale;
  const width = actor.width * camera.scale;
  context.save();
  context.translate(center[0], center[1]);
  context.rotate(-state.heading);
  context.strokeStyle = OBSERVATION_COLOR;
  context.lineWidth = 2;
  context.strokeRect(-length / 2 - 3, -width / 2 - 3, length + 6, width + 6);
  context.restore();
}

const MAP_STYLES: Record<
  MapPolylineKind,
  { color: string; width: number; dash: number[] }
> = {
  lane_centerline: { color: "#b7bac1", width: 1, dash: [6, 6] },
  lane_connector_centerline: { color: "#c9cbd1", width: 1, dash: [3, 6] },
  lane_boundary: { color: "#9ca3af", width: 1, dash: [] },
  road_boundary: { color: "#6b7280", width: 1.8, dash: [] },
  crosswalk: { color: "#b78b48", width: 1.4, dash: [] },
  stop_line: { color: "#929292", width: 2, dash: [] },
};

function drawMap(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  map: SceneMap,
): void {
  context.save();
  context.lineJoin = "round";
  for (const polyline of map.polylines) {
    const style = MAP_STYLES[polyline.kind];
    context.strokeStyle = style.color;
    context.lineWidth = style.width;
    context.setLineDash(style.dash);
    context.beginPath();
    for (let index = 0; index < polyline.points.length; index++) {
      const screen = worldToScreen(polyline.points[index], camera, viewport);
      if (index === 0) context.moveTo(screen[0], screen[1]);
      else context.lineTo(screen[0], screen[1]);
    }
    context.stroke();
  }
  context.restore();
}

function gridInterval(scale: number): number {
  const target = 72 / scale;
  const power = 10 ** Math.floor(Math.log10(target));
  const fraction = target / power;
  return (fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
}

function drawGrid(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
): void {
  const lower = screenToWorld([0, viewport.height], camera, viewport);
  const upper = screenToWorld([viewport.width, 0], camera, viewport);
  const interval = gridInterval(camera.scale);
  context.lineWidth = 1;
  context.strokeStyle = "#f4f4f5";
  context.beginPath();
  for (
    let x = Math.ceil(lower[0] / interval) * interval;
    x <= upper[0];
    x += interval
  ) {
    const screenX = worldToScreen([x, 0], camera, viewport)[0];
    context.moveTo(screenX, 0);
    context.lineTo(screenX, viewport.height);
  }
  for (
    let y = Math.ceil(lower[1] / interval) * interval;
    y <= upper[1];
    y += interval
  ) {
    const screenY = worldToScreen([0, y], camera, viewport)[1];
    context.moveTo(0, screenY);
    context.lineTo(viewport.width, screenY);
  }
  context.stroke();

  const origin = worldToScreen([0, 0], camera, viewport);
  context.strokeStyle = "#e4e4e7";
  context.setLineDash([3, 5]);
  context.beginPath();
  context.moveTo(origin[0], 0);
  context.lineTo(origin[0], viewport.height);
  context.moveTo(0, origin[1]);
  context.lineTo(viewport.width, origin[1]);
  context.stroke();
  context.setLineDash([]);
}

function drawTrails(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  options: DrawOptions,
): void {
  if (!options.frame) return;
  const points = new Map<string, { center: Point; frame: number }[]>();
  const logPoints = new Map<string, { center: Point; frame: number }[]>();
  for (const frame of options.history) {
    if (
      frame.index > options.frame.index ||
      frame.index < options.frame.index - 200
    )
      continue;
    for (const actor of frame.actors) {
      const trail = points.get(actor.id);
      if (trail) trail.push({ center: actor.center, frame: frame.index });
      else points.set(actor.id, [{ center: actor.center, frame: frame.index }]);
      if (actor.log_pose) {
        const logged = { center: actor.log_pose.center, frame: frame.index };
        const logTrail = logPoints.get(actor.id);
        if (logTrail) logTrail.push(logged);
        else logPoints.set(actor.id, [logged]);
      }
    }
  }
  for (const actor of options.caseDetail.actors) {
    const logTrail = logPoints.get(actor.id);
    if (!DEMO && logTrail && actor.control_type && !["log", "static"].includes(actor.control_type)) {
      context.save();
      context.beginPath();
      logTrail.forEach((point, index) => {
        const screen = worldToScreen(point.center, camera, viewport);
        if (index === 0 || point.frame !== logTrail[index - 1].frame + 1)
          context.moveTo(screen[0], screen[1]);
        else context.lineTo(screen[0], screen[1]);
      });
      context.strokeStyle = ACTOR_COLORS[actor.type];
      context.lineWidth = actor.id === options.selectedActorId ? 2 : 1.5;
      context.globalAlpha = actor.id === options.selectedActorId ? 0.7 : 0.4;
      context.setLineDash([6, 4]);
      context.stroke();
      context.restore();
    }
    const trail = points.get(actor.id);
    if (!trail || trail.length < 2) continue;
    context.beginPath();
    for (let index = 0; index < trail.length; index++) {
      const screen = worldToScreen(trail[index].center, camera, viewport);
      if (index === 0 || trail[index].frame !== trail[index - 1].frame + 1)
        context.moveTo(screen[0], screen[1]);
      else context.lineTo(screen[0], screen[1]);
    }
    context.strokeStyle = actorColor(actor);
    context.lineWidth = actor.id === options.selectedActorId ? 2.5 : 1.5;
    context.globalAlpha = actor.id === options.selectedActorId ? 0.8 : 0.35;
    if (DEMO) {
      context.lineWidth = actor.is_ego ? 4 : 1.5;
      context.globalAlpha = actor.is_ego ? 0.9 : 0.3;
    }
    context.stroke();
  }
  context.globalAlpha = 1;
}

function drawActor(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  actor: Actor,
  state: ActorState,
  selected: boolean,
  logged = false,
  override: string | null = null,
): void {
  const pose = logged && state.log_pose ? state.log_pose : state;
  const center = worldToScreen(pose.center, camera, viewport);
  const length = actor.length * camera.scale;
  const width = actor.width * camera.scale;
  const color = logged ? ACTOR_COLORS[actor.type] : override ?? actorColor(actor);
  context.save();
  context.translate(center[0], center[1]);
  context.rotate(-pose.heading);
  if (logged) context.setLineDash([6, 4]);
  if (selected) {
    context.strokeStyle = color;
    context.lineWidth = 1;
    context.setLineDash([4, 3]);
    context.strokeRect(
      -length / 2 - 5,
      -width / 2 - 5,
      length + 10,
      width + 10,
    );
    context.setLineDash([]);
  }
  if (!logged) {
    context.fillStyle = color;
    context.globalAlpha = override ? 0.45 : selected ? 0.22 : 0.12;
    context.fillRect(-length / 2, -width / 2, length, width);
  }
  context.globalAlpha = logged ? 0.55 : 1;
  context.strokeStyle = color;
  context.lineWidth = selected ? 2 : 1.5;
  context.strokeRect(-length / 2, -width / 2, length, width);
  context.beginPath();
  context.moveTo(Math.max(0, length / 2 - 8), -Math.min(width * 0.3, 6));
  context.lineTo(length / 2, 0);
  context.lineTo(Math.max(0, length / 2 - 8), Math.min(width * 0.3, 6));
  context.stroke();
  context.restore();

  if (selected) {
    const reference = worldToScreen(state.position, camera, viewport);
    context.fillStyle = color;
    context.beginPath();
    context.arc(reference[0], reference[1], 2.5, 0, Math.PI * 2);
    context.fill();
  }
}

function drawLabels(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  options: DrawOptions,
  metadata: Map<string, Actor>,
): void {
  if (!options.showLabels) return;
  context.save();
  context.font = CANVAS_FONT;
  context.textBaseline = "middle";
  const labels: ActorLabel[] = [];
  for (const state of frameActors(options.caseDetail, options.frame)) {
    const actor = metadata.get(state.id);
    if (!actor) continue;
    const selected = state.id === options.selectedActorId;
    const bounds = screenBoxBounds(
      state.center,
      state.heading,
      actor.length,
      actor.width,
      camera,
      viewport,
    );
    labels.push({
      id: state.id,
      text: state.id,
      width: context.measureText(state.id).width + 4,
      footprint: selected
        ? [bounds[0] - 5, bounds[1] - 5, bounds[2] + 5, bounds[3] + 5]
        : bounds,
      selected,
    });
  }
  context.lineJoin = "round";
  context.lineWidth = 3;
  context.strokeStyle = "#ffffffdd";
  for (const label of layoutActorLabels(labels, viewport)) {
    const [left, top, , bottom] = label.bounds;
    const y = (top + bottom) / 2;
    context.strokeText(label.text, left + 2, y);
    context.fillStyle = label.selected ? "#1d4ed8" : "#52525b";
    context.fillText(label.text, left + 2, y);
  }
  context.restore();
}

function drawScale(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
): void {
  const interval = gridInterval(camera.scale);
  const width = interval * camera.scale;
  const x = 22;
  const y = viewport.height - 24;
  context.strokeStyle = "#71717a";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(x, y - 4);
  context.lineTo(x, y);
  context.lineTo(x + width, y);
  context.lineTo(x + width, y - 4);
  context.stroke();
  context.fillStyle = "#71717a";
  context.font = CANVAS_FONT;
  context.fillText(`${Number(interval.toPrecision(3))} m`, x, y - 9);
  const axisX = viewport.width - 38;
  const axisY = viewport.height - 33;
  context.beginPath();
  context.moveTo(axisX - 12, axisY);
  context.lineTo(axisX + 12, axisY);
  context.moveTo(axisX, axisY + 12);
  context.lineTo(axisX, axisY - 12);
  context.stroke();
  context.fillText("x", axisX + 17, axisY + 3);
  context.fillText("y", axisX - 3, axisY - 18);
}

export function drawScene(
  context: CanvasRenderingContext2D,
  camera: Camera,
  viewport: Viewport,
  options: DrawOptions,
): void {
  context.clearRect(0, 0, viewport.width, viewport.height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, viewport.width, viewport.height);
  drawGrid(context, camera, viewport);
  if (options.showMap && options.map)
    drawMap(context, camera, viewport, options.map);
  if (options.observation) drawObservedLanes(context, camera, viewport, options.observation);
  const route = options.caseDetail.actors.find((actor) => actor.id === options.selectedActorId)?.route ?? [];
  if (options.showRoute && route.length > 0) {
    context.save();
    context.strokeStyle = "#2563eb";
    context.lineWidth = 2.5;
    context.setLineDash([8, 5]);
    context.beginPath();
    route.forEach((point, index) => {
      const [x, y] = worldToScreen(point, camera, viewport);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    const end = worldToScreen(route[route.length - 1], camera, viewport);
    context.setLineDash([]);
    context.beginPath();
    context.arc(end[0], end[1], 4, 0, 2 * Math.PI);
    context.fillStyle = "#2563eb";
    context.fill();
    context.restore();
  }
  if (options.showTrails) drawTrails(context, camera, viewport, options);
  const metadata = new Map(
    options.caseDetail.actors.map((actor) => [actor.id, actor]),
  );
  const egoEvent = eventColor(options.frame);
  if (options.showTtc) {
    for (const state of frameActors(options.caseDetail, options.frame)) {
      const actor = metadata.get(state.id);
      if (actor?.is_ego)
        drawTtcRange(context, camera, viewport, actor, state,
          options.frame?.ego_events?.includes("ttc") ? EVENT_COLORS.ttc : "#2563eb");
    }
  }
  for (const state of frameActors(options.caseDetail, options.frame)) {
    const actor = metadata.get(state.id);
    if (actor) {
      if (!DEMO && state.log_pose && actor.control_type && !["log", "static"].includes(actor.control_type)) {
        drawActor(context, camera, viewport, actor, state, false, true);
      }
      drawActor(
        context,
        camera,
        viewport,
        actor,
        state,
        state.id === options.selectedActorId,
        false,
        actor.is_ego ? egoEvent : null,
      );
    }
  }
  if (options.observation) {
    const kept = new Set(options.observation.agents);
    for (const state of frameActors(options.caseDetail, options.frame)) {
      const actor = metadata.get(state.id);
      if (actor && kept.has(state.id) && state.id !== options.observation.actor_id)
        drawObservedAgent(context, camera, viewport, actor, state);
    }
  }
  for (const state of options.showGoals ? options.frame?.actors ?? [] : []) {
    const actor = metadata.get(state.id);
    if (!(actor?.is_ego || state.id === options.selectedActorId) || !state.goals?.length) continue;
    const start = worldToScreen(state.position, camera, viewport);
    context.save();
    context.strokeStyle = GOAL_COLOR;
    context.fillStyle = "#fff7ed";
    context.lineWidth = 1.5;
    for (const goal of state.goals) {
      const [x, y] = worldToScreen(goal, camera, viewport);
      context.setLineDash([3, 5]);
      context.beginPath(); context.moveTo(start[0], start[1]); context.lineTo(x, y); context.stroke();
      context.setLineDash([]);
      context.beginPath(); context.arc(x, y, GOAL_RADIUS, 0, 2 * Math.PI);
      context.fill(); context.stroke();
    }
    context.restore();
  }
  for (const item of options.pendingActors ?? []) {
    const [x, y] = worldToScreen(item.position, camera, viewport);
    const length = item.length * camera.scale;
    const width = item.width * camera.scale;
    context.save();
    context.translate(x, y);
    context.rotate(-item.heading);
    context.strokeStyle = "#0f766e";
    context.fillStyle = "#0f766e22";
    context.lineWidth = 1.5;
    context.setLineDash([4, 3]);
    context.fillRect(-length / 2, -width / 2, length, width);
    context.strokeRect(-length / 2, -width / 2, length, width);
    // A tick at the front shows the heading.
    context.setLineDash([]);
    context.beginPath(); context.moveTo(0, 0); context.lineTo(length / 2, 0); context.stroke();
    context.restore();
  }
  // The selected actor's fixed goal: a ring with a solid center, also before it is applied.
  if (options.showGoals && options.pinnedGoal) {
    const [x, y] = worldToScreen(options.pinnedGoal, camera, viewport);
    context.save();
    context.strokeStyle = GOAL_COLOR;
    context.lineWidth = 2;
    context.fillStyle = "#fff7ed";
    context.beginPath(); context.arc(x, y, GOAL_RADIUS, 0, 2 * Math.PI);
    context.fill(); context.stroke();
    context.fillStyle = GOAL_COLOR;
    context.beginPath(); context.arc(x, y, 3, 0, 2 * Math.PI); context.fill();
    context.restore();
  }
  drawLabels(context, camera, viewport, options, metadata);
  drawScale(context, camera, viewport);
}

export function hitTestActor(
  screenPoint: Point,
  camera: Camera,
  viewport: Viewport,
  caseDetail: CaseDetail,
  frame: Frame | null,
): string | null {
  if (!frame) return null;
  const world = screenToWorld(screenPoint, camera, viewport);
  const metadata = new Map(caseDetail.actors.map((actor) => [actor.id, actor]));
  const actors = frameActors(caseDetail, frame);
  for (let index = actors.length - 1; index >= 0; index--) {
    const state = actors[index];
    const actor = metadata.get(state.id);
    if (
      actor &&
      pointInBox(
        world,
        state.center,
        state.heading,
        actor.length,
        actor.width,
        5 / camera.scale,
      )
    ) {
      return state.id;
    }
  }
  return null;
}
