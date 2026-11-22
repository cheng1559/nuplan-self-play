import { memo, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Search, Table2 } from "lucide-react";
import type { WorldMetrics, WorldSummary } from "@/api/types";
import { Input } from "@/components/ui/input";

export const sourceLabels = {
  demo: "Synthetic",
  scene: "Scenario",
  replay: "Replay",
};

type MetricKey = keyof WorldMetrics;
type ColumnKey = "index" | "scenario_id" | "scenario_type" | MetricKey | "dynamic" | "static";

interface Column {
  key: ColumnKey;
  label: string;
  title: string;
  numeric: boolean;
  /** nuPlan multipliers: a zero sends the scene's score to zero. */
  multiplier?: boolean;
}

const COLUMNS: Column[] = [
  { key: "index", label: "#", title: "World index in the batch", numeric: true },
  { key: "scenario_id", label: "Scenario", title: "Scenario id", numeric: false },
  { key: "scenario_type", label: "Type", title: "nuPlan scenario type", numeric: false },
  { key: "score", label: "Score", title: "Ego score (nuPlan-style, training evaluator)", numeric: true },
  { key: "no_ego_at_fault_collisions", label: "Collision", title: "No at-fault collisions (multiplier)", numeric: true, multiplier: true },
  { key: "drivable_area_compliance", label: "Drivable", title: "Drivable area compliance (multiplier)", numeric: true, multiplier: true },
  { key: "ego_is_making_progress", label: "Moving", title: "Ego is making progress (multiplier)", numeric: true, multiplier: true },
  { key: "ego_progress_along_expert_route", label: "Progress", title: "Progress along the expert route (weight 5)", numeric: true },
  { key: "time_to_collision_within_bound", label: "TTC", title: "Time to collision within bound (weight 5)", numeric: true },
  { key: "speed_limit_compliance", label: "Speed", title: "Speed limit compliance (weight 4)", numeric: true },
  { key: "ego_is_comfortable", label: "Comfort", title: "Ego is comfortable (weight 2)", numeric: true },
  { key: "dynamic", label: "Actors", title: "Dynamic actors", numeric: true },
  { key: "static", label: "Static", title: "Static objects", numeric: true },
];

function value(row: WorldSummary, key: ColumnKey): number | string {
  switch (key) {
    case "index":
      return row.index + 1;
    case "scenario_id":
      return row.scenario_id;
    case "scenario_type":
      return row.scenario_type ?? "";
    case "dynamic":
      return row.dynamic_actor_count;
    case "static":
      return row.static_object_count;
    default:
      return row.metrics?.[key] ?? Number.NaN;
  }
}

interface WorldTableProps {
  rows: WorldSummary[];
  skipped: string[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

/** Memoized: the app re-renders every playback frame, the table only when its rows change. */
export const WorldTable = memo(function WorldTable(props: WorldTableProps) {
  const [sort, setSort] = useState<{ key: ColumnKey; ascending: boolean }>({
    key: "index",
    ascending: true,
  });
  const [search, setSearch] = useState("");
  const shown = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const rows = props.rows.filter(
      (row) =>
        !needle ||
        row.scenario_id.toLowerCase().includes(needle) ||
        (row.scenario_type ?? "").toLowerCase().includes(needle),
    );
    const sign = sort.ascending ? 1 : -1;
    return rows.sort((a, b) => {
      const x = value(a, sort.key);
      const y = value(b, sort.key);
      if (typeof x === "string" || typeof y === "string")
        return sign * String(x).localeCompare(String(y)) || a.index - b.index;
      // Missing scores sort last either way.
      if (Number.isNaN(x) || Number.isNaN(y)) return Number(Number.isNaN(x)) - Number(Number.isNaN(y));
      return sign * (x - y) || a.index - b.index;
    });
  }, [props.rows, search, sort]);
  const scored = shown.filter((row) => row.metrics);
  const mean = scored.length
    ? scored.reduce((total, row) => total + (row.metrics?.score ?? 0), 0) / scored.length
    : null;

  const toggle = (key: ColumnKey) =>
    setSort((current) =>
      current.key === key
        ? { key, ascending: !current.ascending }
        : // Scores worst first (to find failures), actor counts largest first.
          { key, ascending: key !== "dynamic" && key !== "static" },
    );

  return (
    <section className="world-table-panel" aria-label="Batch worlds">
      <div className="world-table-heading">
        <Table2 size={16} className="panel-icon" />
        <h2>Batch worlds</h2>
        <span className="count-pill">{shown.length}</span>
        {mean !== null && (
          <span className="world-table-stat">
            mean score <strong>{mean.toFixed(3)}</strong>
          </span>
        )}
        {props.skipped.length > 0 && (
          <span className="world-table-stat" title={props.skipped.join("\n")}>
            {props.skipped.length} skipped (over capacity)
          </span>
        )}
        <div className="world-table-search">
          <Search size={14} aria-hidden="true" />
          <Input
            aria-label="Filter worlds"
            placeholder="Filter by id or type…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
      </div>
      <div className="world-table-scroll">
        <table className="world-table">
          <thead>
            <tr>
              {COLUMNS.map((column) => (
                <th
                  key={column.key}
                  title={column.title}
                  className={column.numeric ? "numeric" : undefined}
                  aria-sort={
                    sort.key === column.key ? (sort.ascending ? "ascending" : "descending") : "none"
                  }
                >
                  <button type="button" onClick={() => toggle(column.key)}>
                    {column.label}
                    {sort.key === column.key &&
                      (sort.ascending ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row) => (
              <tr
                key={row.index}
                className={row.index === props.selectedIndex ? "is-selected" : undefined}
                onClick={() => props.onSelect(row.index)}
                aria-selected={row.index === props.selectedIndex}
              >
                {COLUMNS.map((column) => {
                  const cell = value(row, column.key);
                  if (!column.numeric)
                    return (
                      <td key={column.key} className="text" title={String(cell)}>
                        {cell || "—"}
                      </td>
                    );
                  const number = cell as number;
                  const failed = column.multiplier && number === 0;
                  const metric = column.key in (row.metrics ?? {});
                  return (
                    <td key={column.key} className={`numeric${failed ? " is-failed" : ""}`}>
                      {Number.isNaN(number) ? "—" : metric ? number.toFixed(2) : number}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        {!props.rows.length && (
          <p className="empty-note">Run a batch to list its worlds with their scores.</p>
        )}
      </div>
    </section>
  );
});
