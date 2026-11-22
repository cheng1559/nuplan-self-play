export type Point = readonly [number, number];
export type Bounds = readonly [number, number, number, number];

export interface Camera {
  x: number;
  y: number;
  scale: number;
}

export interface Viewport {
  width: number;
  height: number;
}

export function worldToScreen(point: Point, camera: Camera, viewport: Viewport): Point {
  return [
    viewport.width / 2 + (point[0] - camera.x) * camera.scale,
    viewport.height / 2 - (point[1] - camera.y) * camera.scale,
  ];
}

export function screenToWorld(point: Point, camera: Camera, viewport: Viewport): Point {
  return [
    camera.x + (point[0] - viewport.width / 2) / camera.scale,
    camera.y - (point[1] - viewport.height / 2) / camera.scale,
  ];
}

export function fitCamera(bounds: Bounds, viewport: Viewport, padding = 64): Camera {
  const width = Math.max(bounds[2] - bounds[0], 1);
  const height = Math.max(bounds[3] - bounds[1], 1);
  return {
    x: (bounds[0] + bounds[2]) / 2,
    y: (bounds[1] + bounds[3]) / 2,
    scale: Math.max(
      0.05,
      Math.min(
        Math.max(viewport.width - padding * 2, 1) / width,
        Math.max(viewport.height - padding * 2, 1) / height,
        100,
      ),
    ),
  };
}

export function panCamera(camera: Camera, dx: number, dy: number): Camera {
  return {
    ...camera,
    x: camera.x - dx / camera.scale,
    y: camera.y + dy / camera.scale,
  };
}

export function zoomCamera(
  camera: Camera,
  factor: number,
  anchor: Point,
  viewport: Viewport,
): Camera {
  const world = screenToWorld(anchor, camera, viewport);
  const scale = Math.max(0.05, Math.min(camera.scale * factor, 160));
  return {
    x: world[0] - (anchor[0] - viewport.width / 2) / scale,
    y: world[1] + (anchor[1] - viewport.height / 2) / scale,
    scale,
  };
}

export function pointInBox(
  point: Point,
  center: Point,
  heading: number,
  length: number,
  width: number,
  padding = 0,
): boolean {
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  const localX = dx * Math.cos(heading) + dy * Math.sin(heading);
  const localY = -dx * Math.sin(heading) + dy * Math.cos(heading);
  return Math.abs(localX) <= length / 2 + padding && Math.abs(localY) <= width / 2 + padding;
}
