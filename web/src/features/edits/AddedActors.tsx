import { useState } from "react";
import { Box, CarFront, PersonStanding, Plus, Trash2 } from "lucide-react";
import type { AddedActor, AddedKind, SceneMap } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export const addedKinds: Record<AddedKind, { label: string; length: number; width: number }> = {
  vehicle: { label: "Vehicle", length: 4.8, width: 2.0 },
  pedestrian: { label: "Pedestrian", length: 0.7, width: 0.7 },
  static: { label: "Static object", length: 1.0, width: 1.0 },
};
const icons = { vehicle: CarFront, pedestrian: PersonStanding, static: Box };

/** Direction (rad) of the lane centerline nearest to a point, or 0 without a map. */
export function laneHeading(map: SceneMap | null, [x, y]: [number, number]): number {
  let best = Infinity;
  let heading = 0;
  for (const line of map?.polylines ?? []) {
    if (line.kind !== "lane_centerline" && line.kind !== "lane_connector_centerline") continue;
    for (let index = 1; index < line.points.length; index++) {
      const [ax, ay] = line.points[index - 1];
      const [bx, by] = line.points[index];
      const dx = bx - ax;
      const dy = by - ay;
      const along = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
      const distance = Math.hypot(x - ax - along * dx, y - ay - along * dy);
      if (distance < best) {
        best = distance;
        heading = Math.atan2(dy, dx);
      }
    }
  }
  return heading;
}

interface AddedActorsProps {
  /** The shown scene's added actors, and the serials the shown rollout lacks. */
  items: AddedActor[];
  pending: Set<number>;
  /** Actor ids of the additions the shown rollout contains, by serial. */
  ids: Map<number, string>;
  disabled: boolean;
  /** The kind the next click on the scene places, if any. */
  placing: AddedKind | null;
  onPlace: (kind: AddedKind | null) => void;
  onChange: (serial: number, change: Partial<AddedActor>) => void;
  onRemove: (serial: number) => void;
}

export function AddedActors(props: AddedActorsProps) {
  const [kind, setKind] = useState<AddedKind>("vehicle");
  const number = (
    label: string,
    value: number,
    step: number,
    apply: (value: number) => void,
    serial: number,
  ) => (
    <label>
      <span>{label}</span>
      <Input
        aria-label={`${label} of added actor ${serial}`}
        type="number"
        step={step}
        value={Number(value.toFixed(2))}
        disabled={props.disabled}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (event.target.value !== "" && Number.isFinite(next)) apply(next);
        }}
      />
    </label>
  );
  return (
    <div className="added-actors" aria-label="Added actors">
      <div className="added-actors-new">
        <Select value={kind} disabled={props.disabled} onValueChange={(value) => {
          setKind(value as AddedKind);
          if (props.placing) props.onPlace(value as AddedKind);
        }}>
          <SelectTrigger aria-label="Kind of actor to add" className="min-w-0 flex-1"><SelectValue /></SelectTrigger>
          <SelectContent>
            {(Object.keys(addedKinds) as AddedKind[]).map((value) => (
              <SelectItem key={value} value={value}>{addedKinds[value].label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          variant={props.placing ? "default" : "outline"}
          size="sm"
          aria-pressed={props.placing !== null}
          disabled={props.disabled}
          onClick={() => props.onPlace(props.placing ? null : kind)}
        >
          <Plus size={14} />
          {props.placing ? "Click the scene…" : "Add actor"}
        </Button>
      </div>
      {props.items.map((item) => {
        const Icon = icons[item.kind];
        return (
          <div className="added-actor" key={item.serial}>
            <div className="added-actor-heading">
              <Icon size={14} />
              <span>{addedKinds[item.kind].label}</span>
              {props.ids.has(item.serial) && <span className="added-actor-id">{props.ids.get(item.serial)}</span>}
              {props.pending.has(item.serial) && <Badge variant="outline">Pending</Badge>}
              <Button
                variant="ghost"
                size="xs"
                aria-label={`Remove added actor ${item.serial}`}
                disabled={props.disabled}
                onClick={() => props.onRemove(item.serial)}
              >
                <Trash2 size={13} />
              </Button>
            </div>
            <div className="added-actor-values">
              {number("Heading °", (item.heading * 180) / Math.PI, 5,
                (value) => props.onChange(item.serial, { heading: (value * Math.PI) / 180 }), item.serial)}
              {number("Length m", item.length, 0.1,
                (value) => value > 0 && value <= 30 && props.onChange(item.serial, { length: value }), item.serial)}
              {number("Width m", item.width, 0.1,
                (value) => value > 0 && value <= 30 && props.onChange(item.serial, { width: value }), item.serial)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
