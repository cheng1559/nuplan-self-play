import { useMemo, useRef, useState } from "react";
import { Activity } from "lucide-react";
import type { PolicyTrack } from "@/api/types";

const BLUE = "#2a78d6";
const ORANGE = "#eb6834";
const GRID = "#e4e4e7";
const INK = "#18181b";
const WIDTH = 268;

/** log Gamma (Lanczos), for the Beta density. */
function logGamma(x: number): number {
  const c = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += c[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaDensity(u: number, a: number, b: number) {
  return Math.exp(
    (a - 1) * Math.log(u) + (b - 1) * Math.log(1 - u) -
      (logGamma(a) + logGamma(b) - logGamma(a + b)),
  );
}

function format(value: number | null | undefined, digits = 3) {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

/** Beta density of one action dimension, in physical units, with the applied action. */
function Distribution(props: {
  title: string;
  unit: string;
  range: [number, number];
  alpha: number;
  beta: number;
  applied: number;
}) {
  const height = 64;
  const [low, high] = props.range;
  const points = useMemo(() => {
    const samples = Array.from({ length: 121 }, (_, i) => 0.002 + (0.996 * i) / 120);
    const density = samples.map((u) => betaDensity(u, props.alpha, props.beta));
    const top = Math.max(...density.filter(Number.isFinite), 1e-9);
    return samples.map((u, i) => [
      u * WIDTH,
      height - 2 - (Math.min(density[i], top) / top) * (height - 6),
    ]);
  }, [props.alpha, props.beta]);
  const x = (value: number) => ((value - low) / (high - low)) * WIDTH;
  const path = points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
  return (
    <div className="policy-chart">
      <div className="policy-chart-title">
        <span>{props.title}</span>
        <span>
          applied <strong>{format(props.applied, 2)}</strong> {props.unit}
        </span>
      </div>
      <svg width="100%" viewBox={`0 0 ${WIDTH} ${height + 14}`} role="img"
        aria-label={`${props.title} Beta(${format(props.alpha, 2)}, ${format(props.beta, 2)})`}>
        <line x1={0} x2={WIDTH} y1={height} y2={height} stroke={GRID} />
        {low < 0 && high > 0 && (
          <line x1={x(0)} x2={x(0)} y1={0} y2={height} stroke={GRID} strokeDasharray="2 3" />
        )}
        <path d={`${path}L${WIDTH},${height}L0,${height}Z`} fill={BLUE} fillOpacity={0.12} />
        <path d={path} fill="none" stroke={BLUE} strokeWidth={2} />
        <line x1={x(props.applied)} x2={x(props.applied)} y1={0} y2={height} stroke={INK} strokeWidth={1.5} />
        <text x={0} y={height + 12} className="policy-axis">{low}</text>
        {low < 0 && high > 0 && (
          <text x={x(0)} y={height + 12} textAnchor="middle" className="policy-axis">0</text>
        )}
        <text x={WIDTH} y={height + 12} textAnchor="end" className="policy-axis">{high}</text>
      </svg>
      <div className="policy-chart-note">
        α {format(props.alpha, 2)} · β {format(props.beta, 2)}
      </div>
    </div>
  );
}

interface Series {
  name: string;
  color: string;
  values: (number | null)[];
}

/** Lines over the episode with the current frame; hover reads values, click seeks. */
function TimeChart(props: {
  title: string;
  series: Series[];
  frame: number;
  onSeek: (frame: number) => void;
  details?: (frame: number) => [string, string][];
}) {
  const height = 84;
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const frames = Math.max(...props.series.map((item) => item.values.length), 2);
  const all = props.series.flatMap((item) => item.values.filter((v): v is number => v !== null));
  let low = Math.min(0, ...all);
  let high = Math.max(0, ...all);
  if (high - low < 1e-6) high = low + 1;
  const pad = (high - low) * 0.08;
  low -= pad;
  high += pad;
  const x = (frame: number) => (frame / (frames - 1)) * WIDTH;
  const y = (value: number) => height - ((value - low) / (high - low)) * height;
  const line = (values: (number | null)[]) => {
    let path = "";
    let open = false;
    values.forEach((value, frame) => {
      if (value === null) {
        open = false;
        return;
      }
      path += `${open ? "L" : "M"}${x(frame).toFixed(1)},${y(value).toFixed(1)}`;
      open = true;
    });
    return path;
  };
  const frameAt = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return null;
    const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    return Math.round(ratio * (frames - 1));
  };
  const shown = hover ?? props.frame;
  return (
    <div className="policy-chart">
      <div className="policy-chart-title">
        <span>{props.title}</span>
        {props.series.length > 1 && (
          <span className="policy-legend">
            {props.series.map((item) => (
              <span key={item.name}>
                <i style={{ background: item.color }} />
                {item.name}
              </span>
            ))}
          </span>
        )}
      </div>
      <div className="policy-plot">
        <svg
          ref={ref}
          width="100%"
          viewBox={`0 0 ${WIDTH} ${height}`}
          preserveAspectRatio="none"
          onPointerMove={(event) => setHover(frameAt(event.clientX))}
          onPointerLeave={() => setHover(null)}
          onClick={(event) => {
            const frame = frameAt(event.clientX);
            if (frame !== null) props.onSeek(frame);
          }}
        >
          <line x1={0} x2={WIDTH} y1={y(0)} y2={y(0)} stroke={GRID} />
          {props.series.map((item) => (
            <path key={item.name} d={line(item.values)} fill="none" stroke={item.color}
              strokeWidth={2} vectorEffect="non-scaling-stroke" />
          ))}
          <line x1={x(props.frame)} x2={x(props.frame)} y1={0} y2={height} stroke={INK}
            strokeWidth={1} vectorEffect="non-scaling-stroke" />
          {hover !== null && (
            <line x1={x(hover)} x2={x(hover)} y1={0} y2={height} stroke={INK}
              strokeDasharray="3 3" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          )}
        </svg>
        <span className="policy-ytick policy-ytick-top">{format(high - pad, 2)}</span>
        <span className="policy-ytick policy-ytick-bottom">{format(low + pad, 2)}</span>
      </div>
      <dl className="policy-readout">
        <dt>frame</dt>
        <dd>{shown}</dd>
        {props.series.map((item) => (
          <FragmentPair key={item.name} label={item.name} value={format(item.values[shown])} />
        ))}
        {props.details?.(shown).map(([label, value]) => (
          <FragmentPair key={label} label={label} value={value} />
        ))}
      </dl>
    </div>
  );
}

function FragmentPair({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

/** Discounted return from each frame, per decision (held actions share one decision). */
function returnToGo(policy: PolicyTrack, rewards: (number | null)[]) {
  const frames = rewards.length;
  const starts: number[] = [];
  for (let frame = 0; frame < frames; frame++) {
    const alpha = policy.alpha[frame];
    const previous = frame ? policy.alpha[frame - 1] : null;
    if (alpha && (!previous || previous[0] !== alpha[0] || previous[1] !== alpha[1]))
      starts.push(frame);
  }
  const result: (number | null)[] = new Array(frames).fill(null);
  let future = 0;
  for (let k = starts.length - 1; k >= 0; k--) {
    const end = k + 1 < starts.length ? starts[k + 1] : frames;
    let total = 0;
    for (let frame = starts[k]; frame < end; frame++) total += rewards[frame] ?? 0;
    future = total + policy.gamma * future;
    for (let frame = starts[k]; frame < end; frame++) result[frame] = future;
  }
  return result;
}

const REWARD_PARTS = ["hard", "goal", "ttc", "speed", "comfort", "centerline", "direction", "pace"];

export function PolicyPanel(props: {
  policy: PolicyTrack | null;
  error: string | null;
  frame: number;
  onSeek: (frame: number) => void;
}) {
  const { policy, frame } = props;
  const rewards = policy?.rewards?.reward ?? null;
  const toGo = useMemo(
    () => (policy && rewards ? returnToGo(policy, rewards) : null),
    [policy, rewards],
  );
  // The last frame has no decision: show the one that led to it.
  let last = (policy?.alpha.length ?? 0) - 1;
  while (last >= 0 && !policy?.alpha[last]) last--;
  const decision = Math.min(frame, last);
  const alpha = policy?.alpha[decision];
  const beta = policy?.beta[decision];
  const action = policy?.action[decision];
  return (
    <section className="panel policy-panel">
      <div className="panel-heading">
        <Activity size={17} className="panel-icon" />
        <h2>Policy</h2>
      </div>
      {props.error ? (
        <p className="empty-note">{props.error}</p>
      ) : !policy ? (
        <p className="empty-note">Select a SingleFrameNet actor to see its actions, value, and reward.</p>
      ) : (
        <div className="policy-content">
          {alpha && beta && action && (
            <>
              <Distribution title="Jerk" unit="m/s³" range={policy.jerk_range}
                alpha={alpha[0]} beta={beta[0]} applied={action[0]} />
              <Distribution title="Steering rate" unit="rad/s" range={policy.steering_rate_range}
                alpha={alpha[1]} beta={beta[1]} applied={action[1]} />
            </>
          )}
          <TimeChart
            title="Value"
            series={[
              { name: "value", color: BLUE, values: policy.value },
              ...(toGo ? [{ name: "return-to-go", color: ORANGE, values: toGo }] : []),
            ]}
            frame={frame}
            onSeek={props.onSeek}
          />
          {rewards ? (
            <TimeChart
              title="Reward per step"
              series={[{ name: "reward", color: BLUE, values: rewards }]}
              frame={frame}
              onSeek={props.onSeek}
              details={(at) =>
                REWARD_PARTS.filter((name) => policy.rewards?.[name]).map((name) => [
                  name,
                  format(policy.rewards?.[name]?.[at]),
                ])
              }
            />
          ) : (
            <p className="empty-note">Rewards are scored for the ego only.</p>
          )}
        </div>
      )}
    </section>
  );
}
