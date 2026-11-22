import {
  ChevronFirst,
  ChevronLeft,
  ChevronRight,
  CirclePlay,
  Pause,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface PlaybackControlsProps {
  index: number;
  numFrames: number;
  dt: number;
  playing: boolean;
  speed: number;
  loading: boolean;
  onSeek: (index: number) => void;
  onToggle: () => void;
  onSpeed: (speed: number) => void;
}

export function PlaybackControls({
  index,
  numFrames,
  dt,
  playing,
  speed,
  loading,
  onSeek,
  onToggle,
  onSpeed,
}: PlaybackControlsProps) {
  const last = numFrames - 1;
  return (
    <section className="panel playback-panel" aria-label="Playback controls">
      <div className="panel-heading">
        <CirclePlay size={17} className="panel-icon" />
        <h2>Playback controls</h2>
      </div>
      <div className="playback-content">
        <div className="transport-buttons">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="secondary"
                size="icon"
                aria-label="Restart"
                onClick={() => onSeek(0)}
                disabled={loading || index === 0}
              >
                <ChevronFirst size={17} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Restart</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="secondary"
                size="icon"
                aria-label="Previous frame"
                onClick={() => onSeek(index - 1)}
                disabled={loading || index === 0}
              >
                <ChevronLeft size={18} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Previous frame (←)</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                className="play-button"
                size="icon"
                aria-label={playing ? "Pause" : "Play"}
                onClick={onToggle}
                disabled={loading || numFrames <= 1}
              >
                {playing ? <Pause size={17} /> : <Play size={17} />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {playing ? "Pause (Space)" : "Play (Space)"}
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="secondary"
                size="icon"
                aria-label="Next frame"
                onClick={() => onSeek(index + 1)}
                disabled={loading || index === last}
              >
                <ChevronRight size={18} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Next frame (→)</TooltipContent>
          </Tooltip>
        </div>
        <div className="playback-field">
          <label htmlFor="playback-speed">Playback speed</label>
          <Select
            value={String(speed)}
            onValueChange={(value) => onSpeed(Number(value))}
          >
            <SelectTrigger id="playback-speed" aria-label="Playback speed">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[0.25, 0.5, 1, 2, 4].map((value) => (
                <SelectItem key={value} value={String(value)}>
                  {value}×
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="timeline-labels">
          <span>Frame</span>
          <span className="frame-position">
            <span data-testid="current-frame">{index}</span> / {last}
          </span>
        </div>
        <Slider
          aria-label="Frame"
          min={0}
          max={Math.max(1, last)}
          step={1}
          value={[index]}
          disabled={numFrames <= 1}
          onValueChange={(value) => onSeek(value[0])}
        />
        <div className="playback-position">
          <span className={index === last ? "end-label" : ""}>
            {index === last
              ? "End of recording"
              : loading
                ? "Buffering…"
                : playing
                  ? "Playing"
                  : "Paused"}
          </span>
          <span className="time-position">
            {(index * dt).toFixed(2)}{" "}
            <small>/ {(last * dt).toFixed(2)} s</small>
          </span>
        </div>
      </div>
    </section>
  );
}
