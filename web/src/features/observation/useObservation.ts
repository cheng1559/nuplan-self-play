import { useEffect, useMemo, useRef, useState } from "react";
import { getObservations } from "@/api/client";
import type { Actor, ObservationTrack, ObservationView } from "@/api/types";

/**
 * What the selected SingleFrameNet actor observed at the current frame. The whole
 * episode is fetched once per actor; changing frames reads it locally.
 */
export function useObservation(
  runId: string | null,
  world: number,
  actor: Actor | undefined,
  frame: number,
): { observation: ObservationView | null; error: string | null } {
  const cache = useRef(new Map<string, ObservationTrack>());
  const cachedRun = useRef<string | null>(null);
  const [track, setTrack] = useState<ObservationTrack | null>(null);
  const [error, setError] = useState<string | null>(null);
  const actorId = actor?.control_type === "single_frame" ? actor.id : null;
  const key = runId && world >= 0 && actorId ? `${world}:${actorId}` : null;

  useEffect(() => {
    if (cachedRun.current !== runId) {
      cache.current.clear();
      cachedRun.current = runId;
    }
    setError(null);
    if (!key || !runId || !actorId) {
      setTrack(null);
      return;
    }
    const cached = cache.current.get(key);
    if (cached) {
      setTrack(cached);
      return;
    }
    setTrack(null);
    const controller = new AbortController();
    getObservations(runId, world, actorId, controller.signal)
      .then((result) => {
        cache.current.set(key, result);
        if (!controller.signal.aborted) setTrack(result);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : "Unable to load the observation.");
      });
    return () => controller.abort();
  }, [key, runId, world, actorId]);

  const observation = useMemo(() => {
    const current = track && track.actor_id === actorId ? track.frames[frame] : undefined;
    if (!track || !current) return null;
    const chunks = current.lanes.map((index) => track.chunks[index]);
    return {
      actor_id: track.actor_id,
      frame,
      agents: current.agents,
      agent_capacity: track.agent_capacity,
      lanes: chunks.map((chunk) => chunk.points),
      lane_types: chunks.map((chunk) => chunk.type),
      lane_capacity: track.lane_capacity,
      blind: track.blind,
    };
  }, [track, actorId, frame]);
  return { observation, error };
}
