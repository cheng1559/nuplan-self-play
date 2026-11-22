import type { Actor, ActorState, CaseDetail, Frame } from "./types";

const staticStates = new WeakMap<Actor, ActorState>();

function staticState(actor: Actor): ActorState {
  let state = staticStates.get(actor);
  if (!state) {
    state = {
      id: actor.id, position: actor.static_position!, center: actor.static_position!,
      heading: actor.static_heading ?? 0, velocity: [0, 0], speed: 0,
      acceleration: 0, acceleration_command: 0, steering: 0, yaw_rate: 0,
    };
    staticStates.set(actor, state);
  }
  return state;
}

export function frameActors(detail: CaseDetail, frame: Frame | null): ActorState[] {
  if (!frame) return [];
  const actors = [...frame.actors];
  for (const actor of detail.actors) {
    if (actor.static_position && actor.presence_intervals?.some(
      ([start, stop]) => frame.index >= start && frame.index < stop,
    )) actors.push(staticState(actor));
  }
  return actors;
}

export function findActorState(detail: CaseDetail, frame: Frame | null, id: string | null): ActorState | undefined {
  return frameActors(detail, frame).find((actor) => actor.id === id);
}
