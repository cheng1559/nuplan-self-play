import type React from "react";
import {
  Box,
  CarFront,
  Gauge,
  MapPin,
  MousePointer2,
  PersonStanding,
  Trash2,
  Undo2,
  Users,
} from "lucide-react";
import type {
  Actor,
  ActorState,
  AgentControlKind,
  CaseDetail,
  ControlKind,
  Frame,
  ObservationView,
} from "@/api/types";
import { findActorState } from "@/api/actors";
import { controlLabels, isLearned } from "@/api/controls";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const icons = { vehicle: CarFront, pedestrian: PersonStanding, static: Box };

function agentName(
  actor: Actor,
  detail: CaseDetail,
  frame: Frame | null,
) {
  if (actor.type === "static") return "Static object";
  const applied = actor.control_type ?? "log";
  if (applied === "log")
    return "Log agent";
  if (applied === "absent") return "Absent";
  if (applied === "idm") return actor.has_rail ? "IDM agent · nuPlan R" : "Log fallback";
  if (!actor.route?.length || !actor.route_source || actor.route_source === "none")
    return "Log fallback";
  // The pre-step clock determines which policy produced the displayed frame.
  const transitionTime = Math.max(0, (frame?.index ?? 0) - 1) * detail.dt;
  if (actor.route_source === "gt" &&
    transitionTime > (actor.route_start_s ?? 0) + (actor.route_horizon_s ?? 0) + 1e-4)
    return "Log fallback";
  if (applied === "single_frame") return controlLabels[applied];
  return "Rule pedestrian · oracle";
}

function Values({ actor, state }: { actor: Actor; state: ActorState }) {
  const rows: [string, string, string][] = [
    ["Route", actor.route_source === "map" ? "Map" : actor.route_source === "gt" ? "GT" : "Unavailable", ""],
    ["Speed", state.speed.toFixed(2), "m/s"],
    [
      "Center X / Y",
      `${state.center[0].toFixed(2)} / ${state.center[1].toFixed(2)}`,
      "m",
    ],
    [
      "Reference X / Y",
      `${state.position[0].toFixed(2)} / ${state.position[1].toFixed(2)}`,
      "m",
    ],
    ["Heading", ((state.heading * 180) / Math.PI).toFixed(2), "°"],
    [
      "Velocity X / Y",
      `${state.velocity[0].toFixed(2)} / ${state.velocity[1].toFixed(2)}`,
      "m/s",
    ],
    ["Actual acceleration", state.acceleration.toFixed(3), "m/s²"],
    ["Command acceleration", state.acceleration_command.toFixed(3), "m/s²"],
    ["Tire steering", ((state.steering * 180) / Math.PI).toFixed(2), "°"],
    ["Yaw rate", state.yaw_rate.toFixed(3), "rad/s"],
    [
      "Length × width",
      `${actor.length.toFixed(2)} × ${actor.width.toFixed(2)}`,
      "m",
    ],
  ];
  state.goals?.forEach((goal, index) => {
    // Policies with one goal (the usual case) show it without an index.
    const name = state.goals!.length > 1 ? `Goal ${index + 1}` : "Goal";
    rows.push([`${name} X / Y`, `${goal[0].toFixed(2)} / ${goal[1].toFixed(2)}`, "m"]);
    rows.push([`${name} distance`, Math.hypot(goal[0] - state.position[0], goal[1] - state.position[1]).toFixed(2), "m"]);
  });
  if (actor.route_source === "gt") {
    rows.push(["Route horizon", (actor.route_horizon_s ?? 0).toFixed(1), "s"]);
    if ((actor.route_start_s ?? 0) > 0) rows.push(["Route start", actor.route_start_s!.toFixed(1), "s"]);
  }
  return (
    <dl className="state-values">
      {rows.map(([name, value, unit]) => (
        <div key={name}>
          <dt>{name}</dt>
          <dd>
            {value}
            <span>{unit}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

interface ActorInspectorProps {
  blind: boolean;
  onBlindChange: (blind: boolean) => void;
  controlType: ControlKind | null;
  controlsDisabled: boolean;
  onControlChange: (control: AgentControlKind) => void;
  /** Each actor's control on the next run, to mark deleted actors in the list. */
  controlOf?: (actor: Actor) => ControlKind;
  /** The selected actor's fixed goal for the next run (scene x, y), if any. */
  goal?: [number, number] | null;
  /** True while the next click on the scene places the goal. */
  pickingGoal?: boolean;
  onPickGoal?: () => void;
  onClearGoal?: () => void;
  onDeletedChange?: (deleted: boolean) => void;
  /** The selected actor was added to the scene and has since been removed again. */
  removed?: boolean;
  /** Shown under the actor list (adding actors to the scene). */
  actorsFooter?: React.ReactNode;
  detail: CaseDetail;
  frame: Frame | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** The selected SingleFrameNet actor's observation at this frame. */
  observation?: ObservationView | null;
  observationError?: string | null;
  /** Shown between the actor list and the state table. */
  policyPanel?: React.ReactNode;
}

export function ActorInspector({
  blind,
  onBlindChange,
  controlType,
  controlsDisabled,
  onControlChange,
  controlOf,
  goal = null,
  pickingGoal = false,
  onPickGoal,
  onClearGoal,
  onDeletedChange,
  removed = false,
  actorsFooter,
  detail,
  frame,
  selectedId,
  onSelect,
  observation,
  observationError,
  policyPanel,
}: ActorInspectorProps) {
  const actor = detail.actors.find((item) => item.id === selectedId);
  const state = findActorState(detail, frame, selectedId);
  const applied = actor?.goal_override ?? null;
  const goalPending = goal === null || applied === null
    ? goal !== applied
    : Math.hypot(goal[0] - applied[0], goal[1] - applied[1]) > 1e-3;
  return (
    <aside className="inspector-sidebar">
      <section className="panel actors-panel">
        <div className="panel-heading">
          <Users size={17} className="panel-icon" />
          <h2>Actors</h2>
          <span className="count-pill">{detail.actor_count}</span>
        </div>
        <div className="actor-list" aria-label="Actors">
          {detail.actors.map((item) => {
            const Icon = icons[item.type];
            return (
              <Button
                variant="ghost"
                key={item.id}
                onClick={() => onSelect(item.id)}
                className={`actor-item ${selectedId === item.id ? "is-selected" : ""} ${
                  controlOf?.(item) === "absent" ? "is-absent" : ""}`}
                aria-pressed={selectedId === item.id}
              >
                <span className={`actor-type-icon actor-${item.type}`}>
                  <Icon size={16} />
                </span>
                <span className="actor-id">{item.id}</span>
                <span className="actor-type-name">{item.type}</span>
              </Button>
            );
          })}
        </div>
        {actorsFooter}
      </section>
      {policyPanel}
      <section className="panel inspector-detail">
        <div className="panel-heading">
          <Gauge size={17} className="panel-icon" />
          <h2>Actor state</h2>
        </div>
        {actor ? (
          <>
            <div className="inspector-identity">
              <strong>{actor.id}</strong>
              <Badge variant="outline">{agentName(actor, detail, frame)}</Badge>
            </div>
            {actor.type !== "static" && controlType && (
              <div className="inspector-control">
                <div className="inspector-control-heading">
                  <label htmlFor="actor-control">Control</label>
                  {controlType !== (actor.control_type ?? "log") && (
                    <Badge variant="outline">Pending</Badge>
                  )}
                </div>
                <Select
                  value={controlType}
                  disabled={controlsDisabled}
                  onValueChange={(value) => onControlChange(value as AgentControlKind)}
                >
                  <SelectTrigger id="actor-control" aria-label="Actor control" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="log">{controlLabels.log}</SelectItem>
                    <SelectItem value="single_frame">{controlLabels.single_frame}</SelectItem>
                    <SelectItem value="idm" disabled={!actor.has_rail}>
                      {actor.has_rail ? controlLabels.idm : `${controlLabels.idm} · no rail`}
                    </SelectItem>
                    <SelectItem value="pedestrian">{controlLabels.pedestrian}</SelectItem>
                    <SelectItem value="absent">{controlLabels.absent}</SelectItem>
                  </SelectContent>
                </Select>
                <div className="inspector-control-heading mt-3">
                  <label htmlFor="actor-blind" className="flex items-center gap-2">
                    <Checkbox
                      id="actor-blind"
                      checked={blind}
                      disabled={controlsDisabled}
                      onCheckedChange={(checked) => onBlindChange(checked === true)}
                    />
                    Blind
                  </label>
                  {blind !== (actor.blind ?? false) && (
                    <Badge variant="outline">Pending</Badge>
                  )}
                </div>
                {isLearned(controlType) && (
                  <>
                    <div className="inspector-control-heading mt-3">
                      <label>{goal ? `Goal fixed at ${goal[0].toFixed(1)} / ${goal[1].toFixed(1)} m` : "Goal follows the route"}</label>
                      {goalPending && <Badge variant="outline">Pending</Badge>}
                    </div>
                    <div className="inspector-actions">
                      <Button
                        variant={pickingGoal ? "default" : "outline"}
                        size="sm"
                        aria-pressed={pickingGoal}
                        disabled={controlsDisabled}
                        onClick={onPickGoal}
                      >
                        <MapPin size={14} />
                        {pickingGoal ? "Click the scene…" : goal ? "Move goal" : "Fix goal"}
                      </Button>
                      {goal && (
                        <Button variant="ghost" size="sm" disabled={controlsDisabled} onClick={onClearGoal}>
                          Clear goal
                        </Button>
                      )}
                    </div>
                  </>
                )}
                {removed && <p className="empty-note">Removed from the next run.</p>}
                {!actor.is_ego && !removed && (
                  <div className="inspector-actions mt-3">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={controlsDisabled}
                      onClick={() => onDeletedChange?.(controlType !== "absent")}
                    >
                      {controlType === "absent" ? <Undo2 size={14} /> : <Trash2 size={14} />}
                      {controlType === "absent" ? "Restore actor" : "Delete actor"}
                    </Button>
                  </div>
                )}
              </div>
            )}
            {(observation || observationError) && (
              <div className="inspector-observation" aria-label="Observation">
                <div className="inspector-control-heading">
                  <label>Observation · frame {observation?.frame ?? "…"}</label>
                  {observation?.blind && <Badge variant="outline">Blind</Badge>}
                </div>
                {observation ? (
                  <dl className="state-values">
                    <div>
                      <dt>Agents kept</dt>
                      <dd>
                        {observation.agents.length} / {observation.agent_capacity}
                        <span>incl. self</span>
                      </dd>
                    </div>
                    <div>
                      <dt>Lanes kept</dt>
                      <dd>
                        {observation.lanes.length} / {observation.lane_capacity}
                        <span>segments</span>
                      </dd>
                    </div>
                  </dl>
                ) : (
                  <p className="empty-note">{observationError}</p>
                )}
              </div>
            )}
            {state ? (
              <Values actor={actor} state={state} />
            ) : (
              <p className="empty-note">Not present in this frame.</p>
            )}
          </>
        ) : (
          <div className="inspector-empty">
            <MousePointer2 size={24} />
            <p>Select an actor</p>
            <span>
              Click a footprint in the scene or choose an actor above.
            </span>
          </div>
        )}
      </section>
    </aside>
  );
}
