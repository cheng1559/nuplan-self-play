import { worldToScreen } from "./camera";
import type { Bounds, Camera, Point, Viewport } from "./camera";

export interface ActorLabel {
  id: string;
  text: string;
  footprint: Bounds;
  width: number;
  selected: boolean;
}

export interface PlacedLabel extends ActorLabel {
  bounds: Bounds;
}

const LABEL_HEIGHT = 15;
const GAP = 3;

export function screenBoxBounds(
  center: Point,
  heading: number,
  length: number,
  width: number,
  camera: Camera,
  viewport: Viewport,
): Bounds {
  const point = worldToScreen(center, camera, viewport);
  const cosine = Math.abs(Math.cos(heading));
  const sine = Math.abs(Math.sin(heading));
  const halfX = ((cosine * length + sine * width) * camera.scale) / 2;
  const halfY = ((sine * length + cosine * width) * camera.scale) / 2;
  return [
    point[0] - halfX,
    point[1] - halfY,
    point[0] + halfX,
    point[1] + halfY,
  ];
}

export function boundsOverlap(a: Bounds, b: Bounds, padding = 1): boolean {
  return (
    a[0] < b[2] + padding &&
    a[2] > b[0] - padding &&
    a[1] < b[3] + padding &&
    a[3] > b[1] - padding
  );
}

export function layoutActorLabels(
  labels: ActorLabel[],
  viewport: Viewport,
): PlacedLabel[] {
  const visible = labels.filter(
    ({ footprint: [left, top, right, bottom] }) =>
      right >= 0 &&
      left <= viewport.width &&
      bottom >= 0 &&
      top <= viewport.height,
  );
  const ordered = [...visible].sort(
    (a, b) => Number(b.selected) - Number(a.selected),
  );
  const placed: PlacedLabel[] = [];
  for (const label of ordered) {
    const [left, top, right, bottom] = label.footprint;
    const centerX = (left + right) / 2;
    const centerY = (top + bottom) / 2;
    const candidates: Point[] = [
      [centerX - label.width / 2, top - GAP - LABEL_HEIGHT],
      [centerX - label.width / 2, bottom + GAP],
      [right + GAP, centerY - LABEL_HEIGHT / 2],
      [left - GAP - label.width, centerY - LABEL_HEIGHT / 2],
      [right + GAP, top - GAP - LABEL_HEIGHT],
      [left - GAP - label.width, top - GAP - LABEL_HEIGHT],
      [right + GAP, bottom + GAP],
      [left - GAP - label.width, bottom + GAP],
    ];
    for (const [x, y] of candidates) {
      const bounds: Bounds = [x, y, x + label.width, y + LABEL_HEIGHT];
      if (
        x < GAP ||
        y < GAP ||
        bounds[2] > viewport.width - GAP ||
        bounds[3] > viewport.height - GAP
      )
        continue;
      if (visible.some((other) => boundsOverlap(bounds, other.footprint)))
        continue;
      if (placed.some((other) => boundsOverlap(bounds, other.bounds))) continue;
      placed.push({ ...label, bounds });
      break;
    }
  }
  return placed;
}
