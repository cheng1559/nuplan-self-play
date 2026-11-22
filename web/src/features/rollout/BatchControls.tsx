import { Layers, LoaderCircle, Play } from "lucide-react";
import type {
  ControlPreset,
  LearnedKind,
  PolicyCheckpoint,
  SamplingMethod,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface BatchControlsProps {
  collections: { name: string; count: number }[];
  collection: string;
  onCollection: (name: string) => void;
  collectionSize: number;
  batchSize: number;
  maxBatchSize: number;
  onBatchSize: (size: number) => void;
  batchIndex: number;
  batchCount: number;
  onBatchIndex: (index: number) => void;
  checkpoints: PolicyCheckpoint[];
  checkpoint: string;
  checkpointConfig: string;
  /** Learned family used by the current controls; its checkpoints are listed. */
  learnedKind: LearnedKind | null;
  onCheckpoint: (value: string) => void;
  onCheckpointConfig: (value: string) => void;
  sampling: SamplingMethod;
  onSampling: (value: SamplingMethod) => void;
  goalHorizon: number;
  onGoalHorizon: (value: number) => void;
  selectedCount: number;
  loading: boolean;
  running: boolean;
  preset: ControlPreset;
  custom: boolean;
  onPreset: (value: ControlPreset) => void;
  onRun: () => void;
}

export function BatchControls(props: BatchControlsProps) {
  const disabled = props.loading || props.running;
  const batches = Array.from({ length: props.batchCount }, (_, index) => {
    const first = index * props.batchSize + 1;
    const last = Math.min((index + 1) * props.batchSize, props.collectionSize);
    return { index, label: `${index + 1} · scenes ${first}–${last}` };
  });
  return (
    <section className="panel batch-panel" aria-label="Batch rollout">
      <div className="panel-heading">
        <Layers size={17} className="panel-icon" />
        <h2>Batch rollout</h2>
        <span className="count-pill">{props.selectedCount}</span>
      </div>
      <div className="batch-policy">
        <div className="batch-field">
          <label>Scene set</label>
          <Select value={props.collection} disabled={disabled} onValueChange={props.onCollection}>
            <SelectTrigger aria-label="Scene set" className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              {props.collections.map((item) => (
                <SelectItem key={item.name} value={item.name}>
                  {item.name || "scenes"} · {item.count}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="batch-row">
          <div className="batch-field batch-size">
            <label>Batch size</label>
            <Input
              aria-label="Batch size"
              type="number"
              min={1}
              max={props.maxBatchSize}
              value={props.batchSize}
              disabled={disabled}
              onChange={(event) => props.onBatchSize(Number(event.target.value))}
            />
          </div>
          <div className="batch-field batch-index">
            <label>Batch</label>
            <Select
              value={String(props.batchIndex)}
              disabled={disabled}
              onValueChange={(value) => props.onBatchIndex(Number(value))}
            >
              <SelectTrigger aria-label="Batch" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                {batches.map((item) => (
                  <SelectItem key={item.index} value={String(item.index)}>{item.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="batch-field">
          <label>Controls</label>
          <Select
            value={props.custom ? "custom" : props.preset}
            disabled={disabled}
            onValueChange={(value) => props.onPreset(value as ControlPreset)}
          >
            <SelectTrigger aria-label="Control preset" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {props.custom && <SelectItem value="custom" disabled>Custom</SelectItem>}
              <SelectItem value="ego_single_frame">Ego SingleFrameNet · NR</SelectItem>
              <SelectItem value="ego_single_frame_reactive">Ego SingleFrameNet · R</SelectItem>
              <SelectItem value="single_frame_selfplay">SingleFrameNet self-play</SelectItem>
              <SelectItem value="log">All log (expert)</SelectItem>
              <SelectItem value="ego_idm">Ego IDM</SelectItem>
              <SelectItem value="idm_pedestrian">IDM + pedestrians</SelectItem>
              <SelectItem value="nuplan_reactive">nuPlan reactive</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      {props.learnedKind && <div className="batch-model">
        <label>Checkpoint</label>
        <Select value={props.checkpoints.some((item) => item.path === props.checkpoint) ? props.checkpoint : "custom"}
          disabled={disabled} onValueChange={(value) => {
            const item = props.checkpoints.find((entry) => entry.path === value);
            props.onCheckpoint(item?.path ?? "");
            props.onCheckpointConfig(item?.config_path ?? "");
          }}>
          <SelectTrigger aria-label="Policy checkpoint"><SelectValue /></SelectTrigger>
          <SelectContent>
            {props.checkpoints.map((item) => <SelectItem key={item.path} value={item.path}>{item.name}</SelectItem>)}
            <SelectItem value="custom">Custom checkpoint</SelectItem>
          </SelectContent>
        </Select>
        {!props.checkpoints.some((item) => item.path === props.checkpoint) && <>
          <Input aria-label="Checkpoint path" placeholder="Checkpoint path" value={props.checkpoint}
            disabled={disabled} onChange={(event) => props.onCheckpoint(event.target.value)} />
        </>}
        <label>Action</label>
        <Select value={props.sampling} disabled={disabled}
          onValueChange={(value) => props.onSampling(value as SamplingMethod)}>
          <SelectTrigger aria-label="Action"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="argmax">Beta mode</SelectItem>
            <SelectItem value="expectation">Beta mean</SelectItem>
          </SelectContent>
        </Select>
        {props.learnedKind === "single_frame" && <>
          <label>Goal ahead (s)</label>
          <Input aria-label="Goal horizon" type="number" min={0.5} step={0.5}
            value={props.goalHorizon} disabled={disabled}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (value > 0) props.onGoalHorizon(value);
            }} />
        </>}
      </div>}
      <div className="batch-actions">
        <Button
          size="sm"
          disabled={disabled || !props.selectedCount}
          onClick={props.onRun}
        >
          {props.running ? <LoaderCircle className="animate-spin" /> : <Play />}
          {props.running ? "Running batch…" : `Run ${props.selectedCount} scenes`}
        </Button>
      </div>
    </section>
  );
}
