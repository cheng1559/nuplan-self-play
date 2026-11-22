import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getWorld, listRolloutSources, runBatch } from "@/api/client";
import { usePersistentState } from "@/lib/usePersistentState";
import { isLearned, presetControl, presetLearnedKind } from "@/api/controls";
import type {
  Actor,
  AddedActor,
  PolicyCheckpoint,
  AgentControlKind,
  BatchRollout,
  ControlPreset,
  LearnedKind,
  RolloutOptions,
  RolloutSettings,
  RolloutSource,
  RolloutWorld,
  SamplingMethod,
} from "@/api/types";

function message(cause: unknown) {
  return cause instanceof Error ? cause.message : "Unable to run this batch.";
}

interface ActorSettings {
  control_type: AgentControlKind;
  blind: boolean;
  /** A fixed goal in scene coordinates, replacing the route goal. */
  goal?: [number, number] | null;
}

/** Worlds kept in the browser after viewing, so switching back is instant. */
const WORLD_CACHE = 8;
export const DEFAULT_BATCH_SIZE = 256;

export function useBatchRollout() {
  const [checkpoints, setCheckpoints] = useState<PolicyCheckpoint[]>([]);
  // Each learned family keeps its own checkpoint choice.
  const [selection, setSelection] = useState<Record<LearnedKind, { path: string; config: string }>>({
    single_frame: { path: "", config: "" },
  });
  const [sources, setSources] = useState<RolloutSource[]>([]);
  const [settings, setSettings] = useState<RolloutSettings | null>(null);
  const [collection, setCollection] = useState("");
  const [batchSize, setBatchSizeState] = useState(DEFAULT_BATCH_SIZE);
  const [batchIndex, setBatchIndex] = useState(0);
  const [sampling, setSampling] = useState<SamplingMethod>("argmax");
  // The report evaluates with the route goal 2 s ahead.
  const [goalHorizon, setGoalHorizon] = usePersistentState("rollout.goalHorizon", 2);
  const [batch, setBatch] = useState<BatchRollout | null>(null);
  const [preset, setPreset] = useState<ControlPreset>("ego_single_frame");
  const [overrides, setOverrides] = useState<Record<string, Record<string, ActorSettings>>>({});
  // Stationary actors added to scenes; they stay until removed (presets do not clear them).
  const [added, setAdded] = useState<AddedActor[]>([]);
  // What the shown batch ran with, to tell which additions are still pending.
  const [ranAdded, setRanAdded] = useState<AddedActor[]>([]);
  const serialRef = useRef(0);
  const [worldIndex, setWorldIndex] = useState<number | null>(null);
  const [world, setWorld] = useState<RolloutWorld | null>(null);
  const [worldLoading, setWorldLoading] = useState(false);
  const [loadingSources, setLoadingSources] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sourceError, setSourceError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const activeRef = useRef<AbortController | null>(null);
  const worldCache = useRef(new Map<string, RolloutWorld>());

  const collections = useMemo(() => {
    const counts = new Map<string, number>();
    for (const source of sources) {
      const name = source.collection ?? "";
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    return [...counts].map(([name, count]) => ({ name, count }));
  }, [sources]);
  const members = useMemo(
    () => sources.filter((source) => (source.collection ?? "") === collection),
    [sources, collection],
  );
  const batchCount = Math.max(1, Math.ceil(members.length / batchSize));
  const selectedIds = useMemo(
    () => members.slice(batchIndex * batchSize, (batchIndex + 1) * batchSize).map((s) => s.id),
    [members, batchIndex, batchSize],
  );

  const learnedKind: LearnedKind | null =
    presetLearnedKind(preset) ??
    Object.values(overrides)
      .flatMap((actors) => Object.values(actors).map((settings) => settings.control_type))
      .find(isLearned) ??
    null;
  const checkpoint = learnedKind ? selection[learnedKind].path : "";
  const checkpointConfig = learnedKind ? selection[learnedKind].config : "";
  const setSelected = (field: "path" | "config") => (value: string) => {
    if (!learnedKind) return;
    setSelection((previous) => ({
      ...previous,
      [learnedKind]: { ...previous[learnedKind], [field]: value },
    }));
  };
  const options: RolloutOptions = {
    preset,
    checkpoint: checkpoint || undefined,
    checkpoint_config: checkpointConfig || undefined,
    sampling_method: sampling,
    lazy: true,
    goal_horizon_s: learnedKind === "single_frame" ? goalHorizon : undefined,
    agent_controls: Object.entries(overrides)
      .filter(([source_id]) => selectedIds.includes(source_id))
      .flatMap(([source_id, actors]) =>
        Object.entries(actors).map(([actor_id, { control_type, blind, goal }]) => ({
          source_id,
          actor_id,
          control_type,
          blind,
          ...(goal ? { goal } : {}),
        })),
      ),
  };
  const addedNow = added.filter((item) => selectedIds.includes(item.source_id));
  if (addedNow.length) options.added_actors = addedNow;

  useEffect(() => {
    const controller = new AbortController();
    setLoadingSources(true);
    setSourceError(false);
    setError(null);
    listRolloutSources(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setSources(result.sources);
        setSettings(result.settings);
        setBatchSizeState((size) => Math.min(size, result.settings.max_batch_size));
        setCollection((current) => current || (result.sources[0]?.collection ?? ""));
        const catalog = result.policy_checkpoints ?? [];
        setCheckpoints(catalog);
        setSelection((previous) => {
          const next = { ...previous };
          for (const kind of ["single_frame"] as const) {
            // The newest trained checkpoint, not a random init, when there is one.
            const items = catalog.filter((item) => (item.kind ?? "single_frame") === kind);
            const first = items.find((item) => !item.path.startsWith("random-init")) ?? items[0];
            if (first) next[kind] = { path: first.path, config: first.config_path };
          }
          return next;
        });
        setLoadingSources(false);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(message(cause));
        setSourceError(true);
        setLoadingSources(false);
      });
    return () => {
      controller.abort();
      activeRef.current?.abort();
    };
  }, [attempt]);

  const run = useCallback(() => {
    if (!selectedIds.length || running || loadingSources) return;
    const controller = new AbortController();
    activeRef.current?.abort();
    activeRef.current = controller;
    // A rerun (after editing controls or goals) stays on the world being viewed.
    const viewed = batch?.summaries?.[worldIndex ?? -1]?.source_id;
    setBatch(null);
    setWorld(null);
    setWorldIndex(null);
    worldCache.current.clear();
    setRunning(true);
    setError(null);
    runBatch(selectedIds, controller.signal, options)
      .then((result) => {
        if (controller.signal.aborted || activeRef.current !== controller) return;
        setBatch(result);
        setRanAdded(addedNow);
        const same = result.summaries?.findIndex((item) => item.source_id === viewed) ?? -1;
        setWorldIndex(result.summaries?.length ? Math.max(same, 0) : null);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted && activeRef.current === controller) setError(message(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted && activeRef.current === controller) setRunning(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds, running, loadingSources, batch, worldIndex, JSON.stringify(options)]);

  // Fetch the selected world's frames (or reuse a cached one).
  useEffect(() => {
    if (!batch || worldIndex === null) return;
    const key = `${batch.id}:${worldIndex}`;
    const cached = worldCache.current.get(key);
    if (cached) {
      setWorld(cached);
      return;
    }
    const controller = new AbortController();
    setWorldLoading(true);
    getWorld(batch.id, worldIndex, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        const cache = worldCache.current;
        cache.set(key, result);
        while (cache.size > WORLD_CACHE) cache.delete(cache.keys().next().value as string);
        setWorld(result);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(message(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setWorldLoading(false);
      });
    return () => controller.abort();
  }, [batch, worldIndex]);

  const chooseCollection = (name: string) => {
    setCollection(name);
    setBatchIndex(0);
  };
  const setBatchSize = (size: number) => {
    const limit = settings?.max_batch_size ?? DEFAULT_BATCH_SIZE;
    setBatchSizeState(Math.max(1, Math.min(limit, Math.floor(size) || 1)));
    setBatchIndex(0);
  };

  const applyPreset = (value: ControlPreset) => {
    setPreset(value);
    setOverrides({});
  };

  const controlFor = (sourceId: string, actor: Actor) =>
    overrides[sourceId]?.[actor.id]?.control_type ?? presetControl(preset, actor);

  const blindFor = (sourceId: string, actor: Actor) =>
    overrides[sourceId]?.[actor.id]?.blind ?? false;

  const updateActor = (sourceId: string, actor: Actor, settings: Partial<ActorSettings>) => {
    if (running || actor.type === "static") return;
    setOverrides((previous) => {
      const actors = { ...previous[sourceId] };
      const current = actors[actor.id] ?? {
        control_type: presetControl(preset, actor) as AgentControlKind,
        blind: false,
      };
      const value: ActorSettings = {
        ...current,
        ...settings,
      };
      if (value.control_type === presetControl(preset, actor) && !value.blind && !value.goal)
        delete actors[actor.id];
      else actors[actor.id] = value;
      const next = { ...previous };
      if (Object.keys(actors).length) next[sourceId] = actors;
      else delete next[sourceId];
      return next;
    });
  };

  // The shown world matches the selected row (a stale fetch never shows under another row).
  const shown =
    world && batch && worldIndex !== null && world.source_id === batch.summaries?.[worldIndex]?.source_id
      ? world
      : null;

  return {
    checkpoints: checkpoints.filter((item) => (item.kind ?? "single_frame") === learnedKind),
    checkpoint,
    checkpointConfig,
    setCheckpoint: setSelected("path"),
    setCheckpointConfig: setSelected("config"),
    learnedKind,
    sources,
    settings,
    collections,
    collection,
    chooseCollection,
    collectionSize: members.length,
    batchSize,
    setBatchSize,
    batchIndex,
    setBatchIndex,
    batchCount,
    sampling,
    setSampling,
    goalHorizon,
    setGoalHorizon,
    selectedIds,
    batch,
    preset,
    applyPreset,
    custom: Object.keys(overrides).length > 0,
    controlFor,
    blindFor,
    setActorControl: (sourceId: string, actor: Actor, control: AgentControlKind) =>
      updateActor(sourceId, actor, { control_type: control }),
    setActorBlind: (sourceId: string, actor: Actor, blind: boolean) =>
      updateActor(sourceId, actor, { blind }),
    added,
    /** Additions the shown batch has not run with yet (new or edited since). */
    pendingAdded: added.filter(
      (item) => !ranAdded.some((ran) => JSON.stringify(ran) === JSON.stringify(item)),
    ),
    addActor: (item: Omit<AddedActor, "serial">) => {
      const serial = serialRef.current++;
      setAdded((previous) => [...previous, { ...item, serial }]);
    },
    updateAdded: (sourceId: string, serial: number, change: Partial<AddedActor>) =>
      setAdded((previous) =>
        previous.map((item) =>
          item.source_id === sourceId && item.serial === serial ? { ...item, ...change } : item,
        ),
      ),
    removeAdded: (sourceId: string, serial: number) =>
      setAdded((previous) =>
        previous.filter((item) => !(item.source_id === sourceId && item.serial === serial)),
      ),
    goalFor: (sourceId: string, actor: Actor) => overrides[sourceId]?.[actor.id]?.goal ?? null,
    setActorGoal: (sourceId: string, actor: Actor, goal: [number, number] | null) =>
      updateActor(sourceId, actor, { goal }),
    /** Remove the actor from the next rollout, or bring it back under the preset's control. */
    setActorDeleted: (sourceId: string, actor: Actor, deleted: boolean) => {
      const usual = presetControl(preset, actor) as AgentControlKind;
      updateActor(sourceId, actor, {
        control_type: deleted ? "absent" : usual === "absent" ? "log" : usual,
      });
    },
    world: shown,
    worldIndex,
    worldLoading,
    selectWorld: setWorldIndex,
    loadingSources,
    running,
    error,
    run,
    retry: () => (sourceError ? setAttempt((value) => value + 1) : run()),
  };
}
