import type {
  BatchRollout,
  ObservationTrack,
  PolicyTrack,
  RolloutOptions,
  RolloutSources,
  RolloutWorld,
} from "./types";

const apiRoot = `${import.meta.env.BASE_URL}api/v1`;

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const response = await fetch(`${apiRoot}${path}`, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    if (response.headers.get("content-type")?.includes("application/json")) {
      const body: { detail?: unknown } = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function listRolloutSources(signal?: AbortSignal) {
  return request<RolloutSources>("/rollout-sources", { signal });
}

export function runBatch(
  sourceIds: string[],
  signal?: AbortSignal,
  options: RolloutOptions = {},
) {
  return request<BatchRollout>("/rollouts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_ids: sourceIds, ...options }),
    signal,
  });
}

/** Frames, actors, and map of one world of the latest (lazy) batch. */
export function getWorld(runId: string, world: number, signal?: AbortSignal) {
  return request<RolloutWorld>(`/rollouts/${encodeURIComponent(runId)}/worlds/${world}`, {
    signal,
  });
}

/** Every frame's observation for one actor, in one request. */
export function getObservations(runId: string, world: number, actorId: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ world: String(world), actor_id: actorId });
  return request<ObservationTrack>(`/rollouts/${encodeURIComponent(runId)}/observations?${query}`, {
    signal,
  });
}

/** Action distributions, values, and (ego) rewards of one SingleFrameNet actor. */
export function getPolicy(runId: string, world: number, actorId: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ world: String(world), actor_id: actorId });
  return request<PolicyTrack>(`/rollouts/${encodeURIComponent(runId)}/policy?${query}`, { signal });
}
