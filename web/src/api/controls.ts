import type { Actor, ControlKind, ControlPreset, LearnedKind } from "./types";

const selfplay: Partial<Record<ControlPreset, LearnedKind>> = {
  single_frame_selfplay: "single_frame",
};
const egoOnly: Partial<Record<ControlPreset, ControlKind>> = {
  ego_idm: "idm",
  ego_single_frame: "single_frame",
  ego_single_frame_reactive: "single_frame",
};

export function presetControl(preset: ControlPreset, actor: Actor): ControlKind {
  if (actor.type === "static") return "static";
  // IDM goes only to vehicles nuPlan's reactive builder selected (those with rails).
  if (preset === "idm_pedestrian")
    return actor.type === "vehicle" ? (actor.has_rail ? "idm" : "log") : "pedestrian";
  if (preset === "ego_idm") return actor.is_ego && actor.has_rail ? "idm" : "log";
  if (preset === "ego_single_frame_reactive") {
    // Approximate: the server gives IDM to the 64 railed vehicles nearest the ego.
    if (actor.is_ego) return "single_frame";
    if (actor.type === "pedestrian") return "pedestrian";
    return actor.has_rail ? "idm" : "log";
  }
  if (preset === "nuplan_reactive") {
    if (actor.is_ego) return "log";
    if (actor.reactive_role === "idm") return "idm";
    return actor.reactive_role === "open_loop" ? "log" : "absent";
  }
  const learned = selfplay[preset];
  if (learned) return actor.type === "vehicle" ? learned : "pedestrian";
  return (actor.is_ego && egoOnly[preset]) || "log";
}

export const controlLabels: Record<ControlKind, string> = {
  log: "Log",
  idm: "IDM",
  pedestrian: "Pedestrian · oracle",
  static: "Static",
  single_frame: "SingleFrameNet",
  absent: "Absent",
};

export function isLearned(control: ControlKind): control is LearnedKind {
  return control === "single_frame";
}

/** The learned family a preset implies, if any. */
export function presetLearnedKind(preset: ControlPreset): LearnedKind | null {
  const ego = egoOnly[preset];
  return selfplay[preset] ?? (ego && isLearned(ego) ? ego : null);
}
