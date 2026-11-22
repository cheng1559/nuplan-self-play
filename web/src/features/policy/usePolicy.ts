import { useEffect, useRef, useState } from "react";
import { getPolicy } from "@/api/client";
import type { Actor, PolicyTrack } from "@/api/types";

/** The selected SingleFrameNet actor's whole-episode decisions, fetched once per actor. */
export function usePolicy(runId: string | null, world: number, actor: Actor | undefined) {
  const cache = useRef(new Map<string, PolicyTrack>());
  const cachedRun = useRef<string | null>(null);
  const [track, setTrack] = useState<PolicyTrack | null>(null);
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
    getPolicy(runId, world, actorId, controller.signal)
      .then((result) => {
        cache.current.set(key, result);
        if (!controller.signal.aborted) setTrack(result);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : "Unable to load the policy outputs.");
      });
    return () => controller.abort();
  }, [key, runId, world, actorId]);

  return { policy: track && track.actor_id === actorId ? track : null, error };
}
