import { useEffect, useRef, useState } from "react";
import {
  CarFront,
  CircleHelp,
  FolderOpen,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ActorInspector } from "@/features/inspector/ActorInspector";
import { PlaybackControls } from "@/features/playback/PlaybackControls";
import { useReplay } from "@/features/playback/useReplay";
import { BatchControls } from "@/features/rollout/BatchControls";
import { WorldTable, sourceLabels } from "@/features/rollout/WorldTable";
import { useBatchRollout } from "@/features/rollout/useBatchRollout";
import { BevViewer } from "@/features/viewer/BevViewer";
import { useObservation } from "@/features/observation/useObservation";
import { PolicyPanel } from "@/features/policy/PolicyPanel";
import { usePolicy } from "@/features/policy/usePolicy";
import { isLearned } from "@/api/controls";
import type { Actor, AddedKind } from "@/api/types";
import { AddedActors, addedKinds, laneHeading } from "@/features/edits/AddedActors";

const NONE: never[] = [];

export function App() {
  const rollout = useBatchRollout();
  const replay = useReplay(rollout.world);
  const [selectedActorId, setSelectedActorId] = useState<string | null>(null);
  const inspectedSourceRef = useRef<string | null>(null);
  const detail = replay.detail;
  const selectedActor = detail?.actors.find((actor) => actor.id === selectedActorId);
  const worldIndex = rollout.world ? (rollout.worldIndex ?? -1) : -1;
  const sourceId = rollout.world?.source_id ?? null;
  const controlsDisabled = rollout.running || rollout.loadingSources ||
    !rollout.selectedIds.includes(sourceId ?? "");
  // Edits to the selected actor apply on the next run.
  const editing = selectedActor && sourceId && selectedActor.type !== "static"
    ? { sourceId, actor: selectedActor } : null;
  const control = editing ? rollout.controlFor(editing.sourceId, editing.actor) : null;
  const goal = editing && control && isLearned(control)
    ? rollout.goalFor(editing.sourceId, editing.actor) : null;
  const [pickingGoal, setPickingGoal] = useState(false);
  // The kind of actor the next click on the scene adds.
  const [placing, setPlacing] = useState<AddedKind | null>(null);
  const added = rollout.added.filter((item) => item.source_id === sourceId);
  const pendingAdded = rollout.pendingAdded.filter((item) => item.source_id === sourceId);
  // An added actor that was removed again stays in the shown rollout until the next run.
  const removed = (actor: Actor) =>
    actor.added_serial != null && !added.some((item) => item.serial === actor.added_serial);
  const canDelete = !!editing && !editing.actor.is_ego && !controlsDisabled &&
    !removed(editing.actor);
  const setDeleted = (deleted: boolean) => {
    if (!editing) return;
    if (editing.actor.added_serial != null)
      rollout.removeAdded(editing.sourceId, editing.actor.added_serial);
    else rollout.setActorDeleted(editing.sourceId, editing.actor, deleted);
  };

  useEffect(() => {
    setPickingGoal(false);
  }, [selectedActorId, sourceId, rollout.running]);
  useEffect(() => {
    setPlacing(null);
  }, [sourceId, rollout.running]);
  const { observation, error: observationError } = useObservation(
    rollout.batch?.id ?? null,
    worldIndex,
    selectedActor,
    replay.frameIndex,
  );

  const { policy, error: policyError } = usePolicy(
    rollout.batch?.id ?? null,
    worldIndex,
    selectedActor,
  );

  useEffect(() => {
    if (!detail) return;
    const sameSource = inspectedSourceRef.current === rollout.world?.source_id;
    inspectedSourceRef.current = rollout.world?.source_id ?? null;
    setSelectedActorId((current) => sameSource && detail.actors.some((actor) => actor.id === current)
      ? current : detail.actors[0]?.id ?? null);
  }, [detail?.id]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (
        target &&
        (target.isContentEditable ||
          target.closest(
            '[role="slider"], [role="combobox"], [role="listbox"], [role="option"]',
          ) ||
          ["INPUT", "TEXTAREA"].includes(target.tagName))
      )
        return;
      if (!detail || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key === "Escape" && (pickingGoal || placing)) {
        setPickingGoal(false);
        setPlacing(null);
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && canDelete && editing) {
        event.preventDefault();
        setDeleted(control !== "absent");
        return;
      }
      // A focused button keeps Space and the arrow keys for itself.
      if (target?.tagName === "BUTTON") return;
      if (event.code === "Space") {
        event.preventDefault();
        replay.togglePlay();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        replay.seek(replay.frameIndex + (event.key === "ArrowLeft" ? -1 : 1));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detail, replay.frameIndex, replay.seek, replay.togglePlay, pickingGoal, placing, canDelete,
    editing?.sourceId, editing?.actor, control, rollout.setActorDeleted, rollout.removeAdded]);

  return (
    <div className="app-shell">
      <header className="app-nav">
        <div className="brand">
          <CarFront size={23} />
          <strong>CS5478 Simulation Visualizer</strong>
          <span className="app-description">Batch rollout</span>
        </div>
      </header>
      <div className="workspace">
        <aside className="control-sidebar" aria-label="Scenario controls">
          <BatchControls
            collections={rollout.collections}
            collection={rollout.collection}
            onCollection={rollout.chooseCollection}
            collectionSize={rollout.collectionSize}
            batchSize={rollout.batchSize}
            maxBatchSize={rollout.settings?.max_batch_size ?? 256}
            onBatchSize={rollout.setBatchSize}
            batchIndex={rollout.batchIndex}
            batchCount={rollout.batchCount}
            onBatchIndex={rollout.setBatchIndex}
            sampling={rollout.sampling}
            onSampling={rollout.setSampling}
            goalHorizon={rollout.goalHorizon}
            onGoalHorizon={rollout.setGoalHorizon}
            selectedCount={rollout.selectedIds.length}
            checkpoints={rollout.checkpoints}
            checkpoint={rollout.checkpoint}
            checkpointConfig={rollout.checkpointConfig}
            learnedKind={rollout.learnedKind}
            onCheckpoint={rollout.setCheckpoint}
            onCheckpointConfig={rollout.setCheckpointConfig}
            loading={rollout.loadingSources}
            running={rollout.running}
            preset={rollout.preset}
            custom={rollout.custom}
            onPreset={rollout.applyPreset}
            onRun={rollout.run}
          />
          {detail && (
            <PlaybackControls
              index={replay.frameIndex}
              numFrames={detail.num_frames}
              dt={detail.dt}
              playing={replay.playing}
              speed={replay.speed}
              loading={false}
              onSeek={replay.seek}
              onToggle={replay.togglePlay}
              onSpeed={replay.setSpeed}
            />
          )}
        </aside>
        <main className="analysis-pane">
          <header className="workspace-heading">
            <Tooltip>
              <TooltipTrigger asChild>
                <h1 tabIndex={0}>{detail?.scenario_id ?? "Scenario viewer"}</h1>
              </TooltipTrigger>
              <TooltipContent>
                {detail?.scenario_id ?? "Scenario viewer"}
              </TooltipContent>
            </Tooltip>
            {detail && (
              <div className="case-heading-meta">
                <Badge
                  variant="secondary"
                  className={`source-badge source-${detail.source_kind}`}
                >
                  {sourceLabels[detail.source_kind]}
                </Badge>
                <span>
                  {detail.actor_count} actors <span>·</span>{" "}
                  {(1 / detail.dt).toFixed(0)} Hz
                </span>
              </div>
            )}
          </header>
          <div className="viewer-area">
            {rollout.error ? (
              <div className="workspace-message" role="alert">
                <CircleHelp size={30} />
                <h2>Batch rollout failed</h2>
                <p>{rollout.error}</p>
                <Button onClick={rollout.retry} variant="outline">
                  <RefreshCw size={15} />
                  Retry
                </Button>
              </div>
            ) : rollout.loadingSources || rollout.running ? (
              <div className="workspace-message" role="status">
                <LoaderCircle size={28} className="animate-spin" />
                <h2>
                  {rollout.running
                    ? "Rolling out batch"
                    : "Loading scene sources"}
                </h2>
                <p>
                  {rollout.running
                    ? "Simulating and scoring every world of the batch."
                    : "Reading available normalized scenes."}
                </p>
              </div>
            ) : detail && rollout.world ? (
              <BevViewer
                caseDetail={detail}
                sceneMap={rollout.world.map}
                frame={replay.frame}
                history={replay.history}
                selectedActorId={selectedActorId}
                onSelectActor={setSelectedActorId}
                observation={observation}
                pinnedGoal={goal}
                pendingActors={pendingAdded}
                pickHint={pickingGoal
                  ? "Click the scene to place the goal"
                  : placing
                    ? `Click the scene to place the ${addedKinds[placing].label.toLowerCase()}`
                    : null}
                onPick={(point) => {
                  const position: [number, number] = [
                    Math.round(point[0] * 100) / 100,
                    Math.round(point[1] * 100) / 100,
                  ];
                  if (pickingGoal && editing)
                    rollout.setActorGoal(editing.sourceId, editing.actor, position);
                  else if (placing && sourceId)
                    rollout.addActor({
                      source_id: sourceId,
                      kind: placing,
                      position,
                      // Vehicles line up with the nearest lane.
                      heading: placing === "vehicle" ? laneHeading(rollout.world!.map, position) : 0,
                      length: addedKinds[placing].length,
                      width: addedKinds[placing].width,
                    });
                  setPickingGoal(false);
                  setPlacing(null);
                }}
              />
            ) : rollout.worldLoading ? (
              <div className="workspace-message" role="status">
                <LoaderCircle size={28} className="animate-spin" />
                <h2>Loading world</h2>
              </div>
            ) : (
              <div className="workspace-message">
                <FolderOpen size={32} />
                <h2>No batch loaded</h2>
                <p>
                  Choose a scene set and batch, then run it; pick a world
                  from the table below.
                </p>
              </div>
            )}
          </div>
          <WorldTable
            rows={rollout.batch?.summaries ?? NONE}
            skipped={rollout.batch?.skipped ?? NONE}
            selectedIndex={rollout.worldIndex}
            onSelect={rollout.selectWorld}
          />
        </main>
        {detail ? (
          <ActorInspector
            blind={selectedActor && rollout.world
              ? rollout.blindFor(rollout.world.source_id, selectedActor) : false}
            onBlindChange={(blind) => {
              if (selectedActor && rollout.world)
                rollout.setActorBlind(rollout.world.source_id, selectedActor, blind);
            }}
            controlType={selectedActor && rollout.world
              ? rollout.controlFor(rollout.world.source_id, selectedActor) : null}
            controlsDisabled={controlsDisabled}
            controlOf={(actor) =>
              removed(actor) ? "absent" : sourceId ? rollout.controlFor(sourceId, actor) : "log"}
            goal={goal}
            pickingGoal={pickingGoal}
            onPickGoal={() => {
              setPlacing(null);
              setPickingGoal((value) => !value);
            }}
            removed={!!selectedActor && removed(selectedActor)}
            actorsFooter={
              <AddedActors
                items={added}
                pending={new Set(pendingAdded.map((item) => item.serial))}
                ids={new Map(detail.actors.flatMap((actor) =>
                  actor.added_serial != null ? [[actor.added_serial, actor.id]] : []))}
                disabled={controlsDisabled}
                placing={placing}
                onPlace={(kind) => {
                  setPickingGoal(false);
                  setPlacing(kind);
                }}
                onChange={(serial, change) => {
                  if (sourceId) rollout.updateAdded(sourceId, serial, change);
                }}
                onRemove={(serial) => {
                  if (sourceId) rollout.removeAdded(sourceId, serial);
                }}
              />
            }
            onClearGoal={() => {
              if (editing) rollout.setActorGoal(editing.sourceId, editing.actor, null);
            }}
            onDeletedChange={setDeleted}
            onControlChange={(control) => {
              if (selectedActor && rollout.world)
                rollout.setActorControl(rollout.world.source_id, selectedActor, control);
            }}
            detail={detail}
            frame={replay.frame}
            selectedId={selectedActorId}
            onSelect={setSelectedActorId}
            observation={observation}
            observationError={observationError}
            policyPanel={
              <PolicyPanel
                policy={policy}
                error={policyError}
                frame={replay.frameIndex}
                onSeek={replay.seek}
              />
            }
          />
        ) : (
          <aside className="inspector-sidebar">
            <section className="panel">
              <div className="panel-heading">
                <FolderOpen size={17} className="panel-icon" />
                <h2>Actor state</h2>
              </div>
              <div className="inspector-empty">
                <p>Complete a rollout to inspect actor states.</p>
              </div>
            </section>
          </aside>
        )}
      </div>
    </div>
  );
}
