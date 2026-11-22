import { useCallback, useEffect, useMemo, useState } from "react";
import type { RolloutWorld } from "@/api/types";

interface PlaybackState {
  worldId: string | null;
  index: number;
  playing: boolean;
}

export function useReplay(world: RolloutWorld | null) {
  const worldId = world?.detail.id ?? null;
  const [state, setState] = useState<PlaybackState>({
    worldId: null,
    index: 0,
    playing: false,
  });
  const [speed, setSpeed] = useState(1);
  const current: PlaybackState =
    state.worldId === worldId ? state : { worldId, index: 0, playing: false };
  const last = (world?.frames.length ?? 1) - 1;
  const frameIndex = Math.min(current.index, last);
  const frame = world?.frames[frameIndex] ?? null;
  const history = useMemo(
    () =>
      world?.frames.slice(Math.max(0, frameIndex - 119), frameIndex + 1) ?? [],
    [world, frameIndex],
  );

  useEffect(() => {
    if (!world || !current.playing || frameIndex >= last) return;
    const timer = window.setTimeout(
      () => {
        setState((previous) =>
          previous.worldId !== worldId
            ? previous
            : {
                ...previous,
                index: Math.min(last, previous.index + 1),
                playing: previous.index + 1 < last,
              },
        );
      },
      (world.detail.dt * 1000) / speed,
    );
    return () => window.clearTimeout(timer);
  }, [world, worldId, current.playing, frameIndex, last, speed]);

  const seek = useCallback(
    (index: number) => {
      if (!world) return;
      setState({
        worldId,
        index: Math.max(0, Math.min(last, index)),
        playing: false,
      });
    },
    [world, worldId, last],
  );

  const togglePlay = useCallback(() => {
    if (!world || last <= 0) return;
    setState((previous) => {
      const active =
        previous.worldId === worldId
          ? previous
          : { worldId, index: 0, playing: false };
      return {
        worldId,
        index: active.index === last ? 0 : active.index,
        playing: !active.playing,
      };
    });
  }, [world, worldId, last]);

  return {
    detail: world?.detail ?? null,
    frame,
    frameIndex,
    history,
    playing: current.playing,
    speed,
    setSpeed,
    seek,
    togglePlay,
  };
}
