import {
  Crosshair,
  Eye,
  Map as MapIcon,
  Maximize2,
  Route,
  Navigation2,
  Target,
  Tags,
  Timer,
} from "lucide-react";
import { useCallback, useEffect, useRef } from "react";
import { usePersistentState } from "@/lib/usePersistentState";

/** Zoom chosen with the wheel, kept across worlds (and reloads); the Fit button clears it. */
const ZOOM_KEY = "viewer.zoom";
function savedZoom(): number | null {
  try {
    const value = Number(window.localStorage.getItem(ZOOM_KEY));
    return value > 0 ? value : null;
  } catch {
    return null;
  }
}
function saveZoom(value: number | null) {
  try {
    if (value === null) window.localStorage.removeItem(ZOOM_KEY);
    else window.localStorage.setItem(ZOOM_KEY, String(value));
  } catch {
    // Storage unavailable: the zoom simply resets per world.
  }
}
import type { PointerEvent as ReactPointerEvent } from "react";
import type { CaseDetail, Frame, ObservationView, SceneMap } from "@/api/types";
import { Button } from "@/components/ui/button";
import { fitCamera, panCamera, screenToWorld, zoomCamera } from "@/rendering/camera";
import type { Camera, Point, Viewport } from "@/rendering/camera";
import { drawScene, hitTestActor } from "@/rendering/draw";
import { findActorState } from "@/api/actors";
import "./viewer.css";

export interface BevViewerProps {
  caseDetail: CaseDetail;
  sceneMap: SceneMap | null;
  frame: Frame | null;
  history: Frame[];
  selectedActorId: string | null;
  onSelectActor: (id: string | null) => void;
  /** What the selected SingleFrameNet actor kept at this frame. */
  observation?: ObservationView | null;
  /** The selected actor's fixed goal in scene coordinates, applied or pending. */
  pinnedGoal?: Point | null;
  /** Added actors not yet in the shown rollout, as dashed outlines. */
  pendingActors?: { position: Point; heading: number; length: number; width: number }[];
  /** While set, the next click on the scene reports its point; the text is the hint. */
  pickHint?: string | null;
  onPick?: (point: Point) => void;
}

interface PointerDrag {
  start: Point;
  previous: Point;
  moved: boolean;
}

export function BevViewer(props: BevViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<Camera>({ x: 0, y: 0, scale: 10 });
  const viewportRef = useRef<Viewport>({ width: 1, height: 1 });
  const pendingDrawRef = useRef<number | null>(null);
  const fittedRef = useRef(false);
  const dragRef = useRef<PointerDrag | null>(null);
  const followOffsetRef = useRef<Point>([0, 0]);
  // Toolbar choices outlive the viewer, which remounts for every world and batch.
  const [followActor, setFollowActor] = usePersistentState("viewer.follow", false);
  const [showTrails, setShowTrails] = usePersistentState("viewer.trails", true);
  const [showLabels, setShowLabels] = usePersistentState("viewer.labels", true);
  const [showRoute, setShowRoute] = usePersistentState("viewer.route", true);
  const [showGoals, setShowGoals] = usePersistentState("viewer.goals", true);
  const [showMap, setShowMap] = usePersistentState("viewer.map", true);
  const [showObservation, setShowObservation] = usePersistentState("viewer.observation", true);
  const [showTtc, setShowTtc] = usePersistentState("viewer.ttc", true);
  const latestRef = useRef({
    ...props,
    followActor,
    showTrails,
    showLabels,
    showMap,
    showRoute,
    showGoals,
    showTtc,
    map: props.sceneMap,
    observation: showObservation ? props.observation : null,
  });
  latestRef.current = {
    ...props,
    followActor,
    showTrails,
    showLabels,
    showMap,
    showRoute,
    showGoals,
    showTtc,
    map: props.sceneMap,
    observation: showObservation ? props.observation : null,
  };

  const effectiveCamera = useCallback((): Camera => {
    const latest = latestRef.current;
    const selected = latest.followActor
      ? findActorState(latest.caseDetail, latest.frame, latest.selectedActorId)
      : null;
    return selected
      ? {
          ...cameraRef.current,
          x: selected.center[0] + followOffsetRef.current[0],
          y: selected.center[1] + followOffsetRef.current[1],
        }
      : cameraRef.current;
  }, []);

  const setCamera = useCallback((camera: Camera) => {
    const latest = latestRef.current;
    const selected = latest.followActor
      ? findActorState(latest.caseDetail, latest.frame, latest.selectedActorId)
      : null;
    if (selected) {
      followOffsetRef.current = [
        camera.x - selected.center[0],
        camera.y - selected.center[1],
      ];
    }
    cameraRef.current = camera;
  }, []);

  const scheduleDraw = useCallback(() => {
    if (pendingDrawRef.current !== null) return;
    pendingDrawRef.current = requestAnimationFrame(() => {
      pendingDrawRef.current = null;
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (!canvas || !context) return;
      const ratio = window.devicePixelRatio || 1;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      drawScene(
        context,
        effectiveCamera(),
        viewportRef.current,
        latestRef.current,
      );
    });
  }, [effectiveCamera]);

  useEffect(() => {
    let active = true;
    void document.fonts.ready.then(() => {
      if (active) scheduleDraw();
    });
    return () => {
      active = false;
    };
  }, [scheduleDraw]);

  // Automatic fits (new scene, first layout) keep Follow; the Fit button turns it off.
  const fitScene = useCallback((keepFollow = false) => {
    if (viewportRef.current.width <= 1 || viewportRef.current.height <= 1)
      return;
    const zoom = keepFollow ? savedZoom() : null;
    if (!keepFollow) saveZoom(null);
    const { caseDetail, frame } = latestRef.current;
    const ego = caseDetail.actors.find((actor) => actor.is_ego)?.id;
    const center = frame?.actors.find((actor) => actor.id === ego)?.center;
    cameraRef.current =
      zoom !== null && center
        ? { x: center[0], y: center[1], scale: zoom }
        : fitCamera(caseDetail.bounds, viewportRef.current);
    followOffsetRef.current = [0, 0];
    if (!keepFollow) {
      latestRef.current.followActor = false;
      setFollowActor(false);
    }
    fittedRef.current = true;
    scheduleDraw();
  }, [scheduleDraw]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (!width || !height) return;
      viewportRef.current = { width, height };
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      if (!fittedRef.current) fitScene(true);
      else scheduleDraw();
    });
    observer.observe(wrapper);
    return () => {
      observer.disconnect();
      if (pendingDrawRef.current !== null)
        cancelAnimationFrame(pendingDrawRef.current);
      pendingDrawRef.current = null;
    };
  }, [fitScene, scheduleDraw]);

  useEffect(() => {
    fittedRef.current = false;
    fitScene(true);
  }, [props.caseDetail.id, fitScene]);

  useEffect(() => {
    followOffsetRef.current = [0, 0];
  }, [props.selectedActorId]);

  useEffect(() => {
    if (canvasRef.current) canvasRef.current.style.cursor = props.pickHint ? "crosshair" : "grab";
  }, [props.pickHint]);

  useEffect(() => {
    scheduleDraw();
  }, [
    props.frame,
    props.history,
    props.selectedActorId,
    followActor,
    showTrails,
    showLabels,
    showMap,
    showRoute,
    showGoals,
    showTtc,
    props.sceneMap,
    props.observation,
    props.pinnedGoal,
    props.pendingActors,
    showObservation,
    scheduleDraw,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const anchor: Point = [
        event.clientX - rect.left,
        event.clientY - rect.top,
      ];
      const zoomed = zoomCamera(
        effectiveCamera(),
        Math.exp(-Math.max(-100, Math.min(event.deltaY, 100)) * 0.003),
        anchor,
        viewportRef.current,
      );
      setCamera(zoomed);
      saveZoom(zoomed.scale);
      scheduleDraw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [effectiveCamera, scheduleDraw, setCamera]);

  const localPoint = (event: ReactPointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    const point = localPoint(event);
    dragRef.current = { start: point, previous: point, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = localPoint(event);
    const drag = dragRef.current;
    if (drag) {
      if (Math.hypot(point[0] - drag.start[0], point[1] - drag.start[1]) > 4)
        drag.moved = true;
      if (drag.moved) {
        setCamera(
          panCamera(
            effectiveCamera(),
            point[0] - drag.previous[0],
            point[1] - drag.previous[1],
          ),
        );
        event.currentTarget.style.cursor = "grabbing";
        scheduleDraw();
      }
      drag.previous = point;
    } else if (latestRef.current.pickHint) {
      event.currentTarget.style.cursor = "crosshair";
    } else {
      const latest = latestRef.current;
      event.currentTarget.style.cursor = hitTestActor(
        point,
        effectiveCamera(),
        viewportRef.current,
        latest.caseDetail,
        latest.frame,
      )
        ? "pointer"
        : "grab";
    }
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag && !drag.moved && latestRef.current.pickHint) {
      latestRef.current.onPick?.(
        screenToWorld(localPoint(event), effectiveCamera(), viewportRef.current),
      );
    } else if (drag && !drag.moved) {
      const latest = latestRef.current;
      const actorId = hitTestActor(
        localPoint(event),
        effectiveCamera(),
        viewportRef.current,
        latest.caseDetail,
        latest.frame,
      );
      if (actorId === null && latest.followActor) {
        cameraRef.current = effectiveCamera();
        latest.followActor = false;
        setFollowActor(false);
      }
      latest.onSelectActor(actorId);
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    event.currentTarget.style.cursor = latestRef.current.pickHint ? "crosshair" : "grab";
  };

  return (
    <div className="bev-viewer">
      <div className="bev-toolbar" role="toolbar" aria-label="View controls">
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          onClick={() => fitScene()}
          aria-label="Fit scene"
        >
          <Maximize2 size={14} />
          <span>Fit</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Follow actor"
          aria-pressed={followActor}
          disabled={!props.selectedActorId}
          onClick={() => {
            const enabled = !latestRef.current.followActor;
            if (!enabled) cameraRef.current = effectiveCamera();
            followOffsetRef.current = [0, 0];
            latestRef.current.followActor = enabled;
            setFollowActor(enabled);
            scheduleDraw();
          }}
        >
          <Crosshair size={14} />
          <span>Follow</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show route"
          aria-pressed={showRoute}
          onClick={() => setShowRoute(!showRoute)}
        >
          <Navigation2 size={14} />
          <span>Route</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show goals"
          aria-pressed={showGoals}
          onClick={() => setShowGoals(!showGoals)}
        >
          <Target size={14} />
          <span>Goal</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show TTC range"
          aria-pressed={showTtc}
          onClick={() => setShowTtc(!showTtc)}
        >
          <Timer size={14} />
          <span>TTC</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show observation"
          aria-pressed={showObservation}
          disabled={!props.observation}
          onClick={() => setShowObservation(!showObservation)}
        >
          <Eye size={14} />
          <span>Obs</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show trails"
          aria-pressed={showTrails}
          onClick={() => setShowTrails(!showTrails)}
        >
          <Route size={14} />
          <span>Trails</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show map"
          aria-pressed={showMap}
          disabled={!props.sceneMap}
          onClick={() => setShowMap(!showMap)}
        >
          <MapIcon size={14} />
          <span>Map</span>
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bev-tool"
          aria-label="Show labels"
          aria-pressed={showLabels}
          onClick={() => setShowLabels(!showLabels)}
        >
          <Tags size={14} />
          <span>Labels</span>
        </Button>
      </div>
      <div className="bev-canvas-wrapper" ref={wrapperRef}>
        <canvas
          ref={canvasRef}
          className="bev-canvas"
          data-testid="bev-canvas"
          aria-label="Scenario bird’s-eye view"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={(event) => {
            dragRef.current = null;
            event.currentTarget.style.cursor = "grab";
          }}
        />
        {!props.frame && (
          <div className="bev-loading" role="status">
            Loading frame…
          </div>
        )}
        {props.pickHint && (
          <div className="bev-pick-hint" role="status">
            {props.pickHint} · Esc cancels
          </div>
        )}
      </div>
      <div className="bev-info">
        <div className="bev-map-status" data-testid="map-status" role="status">
          {props.sceneMap
            ? `${props.sceneMap.map_name}${showMap ? "" : " · hidden"}`
            : "No map available"}
        </div>
      </div>
      {props.sceneMap && showMap && (
        <div className="bev-map-legend" aria-label="Map legend">
          <span>
            <i className="bev-line bev-lane-boundary" /> Lane boundary
          </span>
          <span>
            <i className="bev-line bev-centerline" /> Centerline
          </span>
          {props.sceneMap.polylines.some(
            (line) => line.kind === "road_boundary",
          ) && (
            <span>
              <i className="bev-line bev-road-boundary" /> Drivable outline
            </span>
          )}
          {props.sceneMap.polylines.some(
            (line) => line.kind === "crosswalk",
          ) && (
            <span>
              <i className="bev-line bev-crosswalk" /> Crosswalk
            </span>
          )}
        </div>
      )}
      <div className="bev-legend" aria-label="Actor legend">
        <span>
          <i className="bev-dot" style={{ background: "#2563eb" }} /> Controlled / Ego
        </span>
        <span>
          <i className="bev-dot" style={{ background: "#dc2626" }} /> Blind
        </span>
        <span>
          <i className="bev-dot" style={{ background: "#dc2626" }} /> Ego collision
        </span>
        <span>
          <i className="bev-dot" style={{ background: "#db2777" }} /> Ego off-road
        </span>
        <span>
          <i className="bev-dot" style={{ background: "#f59e0b" }} /> Ego TTC &lt; 0.95 s
        </span>
        <span>
          <i className="bev-dot bev-vehicle" /> Vehicle
        </span>
        <span>
          <i className="bev-dot bev-pedestrian" /> Pedestrian
        </span>
        <span>
          <i className="bev-dot bev-static" /> Static object
        </span>
      </div>
    </div>
  );
}
