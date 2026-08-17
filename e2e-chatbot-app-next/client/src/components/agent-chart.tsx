import { memo, useMemo } from 'react';
import { cn } from '@/lib/utils';
import { toNumber, type ChartSpec, type ChartSeries } from '@/lib/chart-spec';

const CHART_WIDTH = 720;
const CHART_HEIGHT = 360;
const SERIES_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
];

/** Axis labels stay short; tooltips carry the exact figure. */
const compactFormat = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  maximumFractionDigits: 1,
});
const fullFormat = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
});

/**
 * Past the end of the palette, hues repeat — so each extra lap is blended
 * toward the background, keeping a seventh slice distinct from the first.
 */
const seriesColor = (index: number) => {
  const base = SERIES_COLORS[index % SERIES_COLORS.length];
  const lap = Math.floor(index / SERIES_COLORS.length);
  if (lap === 0) return base;
  const strength = Math.max(30, 100 - lap * 35);
  return `color-mix(in srgb, ${base} ${strength}%, var(--background))`;
};

/** Approximate width of one character at the 11px label size. */
const LABEL_CHAR_WIDTH = 5.6;
const LABEL_MAX_CHARS = 24;
const LABEL_ROTATION_DEGREES = 35;
const LABEL_ROTATION_SIN = Math.sin((LABEL_ROTATION_DEGREES * Math.PI) / 180);

const truncateLabel = (label: string) =>
  label.length > LABEL_MAX_CHARS
    ? `${label.slice(0, LABEL_MAX_CHARS - 1)}…`
    : label;

type Scale = { min: number; max: number; ticks: number[] };

/** Round a tick to a value that survives floating-point accumulation. */
const cleanTick = (value: number) => Number(value.toFixed(10));

function niceScale(dataMin: number, dataMax: number, tickCount = 5): Scale {
  // Bars and areas are only honest when measured from zero.
  const min = Math.min(0, dataMin);
  let max = Math.max(0, dataMax);
  if (min === max) max = min + 1;

  const rawStep = (max - min) / tickCount;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step =
    (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) *
    magnitude;

  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = niceMin; value <= niceMax + step * 1e-9; value += step) {
    ticks.push(cleanTick(value));
  }

  return { min: niceMin, max: niceMax, ticks };
}

const categoryLabel = (row: Record<string, unknown>, xKey: string) =>
  String(row[xKey] ?? '');

function ChartLegend({ series }: { series: ChartSeries[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 pb-3 text-muted-foreground text-xs">
      {series.map((entry, index) => (
        <span key={entry.key} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="size-2.5 shrink-0 rounded-[2px]"
            style={{ backgroundColor: seriesColor(index) }}
          />
          {entry.label}
        </span>
      ))}
    </div>
  );
}

function CartesianChart({ spec }: { spec: ChartSpec }) {
  const { type, xKey, series, data } = spec;
  const horizontal = type === 'horizontalBar';

  const values = data.flatMap((row) =>
    series.map((entry) => toNumber(row[entry.key])).filter((v) => v !== null),
  ) as number[];
  const scale = niceScale(Math.min(...values), Math.max(...values));

  const labels = data.map((row) => categoryLabel(row, xKey));
  const shown = labels.map(truncateLabel);
  const longestLabel = Math.max(...shown.map((label) => label.length));
  const labelWidth = longestLabel * LABEL_CHAR_WIDTH;
  // Rotated labels trail down and to the left of their anchor, so the axis has
  // to reserve their vertical reach or the longest one gets clipped away.
  const rotateLabels = longestLabel > 6;

  const margin = horizontal
    ? { top: 8, right: 32, bottom: 40, left: 20 + labelWidth }
    : {
        top: 8,
        right: 16,
        bottom: rotateLabels ? 28 + labelWidth * LABEL_ROTATION_SIN : 44,
        left: 64,
      };

  const plotWidth = CHART_WIDTH - margin.left - margin.right;
  const plotHeight = CHART_HEIGHT - margin.top - margin.bottom;

  // Position along the value axis.
  const valueLength = horizontal ? plotWidth : plotHeight;
  const valueAt = (value: number) => {
    const ratio = (value - scale.min) / (scale.max - scale.min);
    return horizontal ? ratio * valueLength : valueLength - ratio * valueLength;
  };
  const zero = valueAt(0);

  // Position along the category axis.
  const band = (horizontal ? plotHeight : plotWidth) / data.length;
  const bandPadding = band * 0.15;
  const groupWidth = band - bandPadding * 2;
  const barWidth = groupWidth / series.length;
  const bandCenter = (index: number) => index * band + band / 2;

  // Crowded category axes drop labels rather than overlapping them.
  const labelStride = horizontal
    ? Math.ceil(data.length / 18)
    : Math.ceil(data.length / 14);

  return (
    <svg
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      className="h-auto w-full"
      role="img"
      aria-label={spec.title ?? 'Chart'}
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform={`translate(${margin.left} ${margin.top})`}>
        {/* Gridlines and value-axis ticks */}
        {scale.ticks.map((tick) => {
          const offset = valueAt(tick);
          return horizontal ? (
            <g key={tick}>
              <line
                x1={offset}
                x2={offset}
                y1={0}
                y2={plotHeight}
                stroke="currentColor"
                className="text-border"
                strokeWidth={1}
              />
              <text
                x={offset}
                y={plotHeight + 20}
                textAnchor="middle"
                className="fill-muted-foreground text-[11px]"
              >
                {compactFormat.format(tick)}
              </text>
            </g>
          ) : (
            <g key={tick}>
              <line
                x1={0}
                x2={plotWidth}
                y1={offset}
                y2={offset}
                stroke="currentColor"
                className="text-border"
                strokeWidth={1}
              />
              <text
                x={-10}
                y={offset + 4}
                textAnchor="end"
                className="fill-muted-foreground text-[11px]"
              >
                {compactFormat.format(tick)}
              </text>
            </g>
          );
        })}

        {/* Category labels */}
        {shown.map((label, index) => {
          if (index % labelStride !== 0) return null;
          const center = bandCenter(index);
          const full = labels[index];
          const baseline = plotHeight + 18;

          return horizontal ? (
            <text
              key={`label-${center}`}
              x={-10}
              y={center + 4}
              textAnchor="end"
              className="fill-muted-foreground text-[11px]"
            >
              {label}
              {label !== full && <title>{full}</title>}
            </text>
          ) : (
            <text
              key={`label-${center}`}
              x={center}
              y={baseline}
              textAnchor={rotateLabels ? 'end' : 'middle'}
              transform={
                rotateLabels
                  ? `rotate(-${LABEL_ROTATION_DEGREES} ${center} ${baseline})`
                  : undefined
              }
              className="fill-muted-foreground text-[11px]"
            >
              {label}
              {label !== full && <title>{full}</title>}
            </text>
          );
        })}

        {/* Baseline */}
        <line
          x1={horizontal ? zero : 0}
          x2={horizontal ? zero : plotWidth}
          y1={horizontal ? 0 : zero}
          y2={horizontal ? plotHeight : zero}
          stroke="currentColor"
          className="text-muted-foreground/50"
          strokeWidth={1}
        />

        {series.map((entry, seriesIndex) => {
          const color = seriesColor(seriesIndex);
          const points = data
            .map((row, index) => {
              const value = toNumber(row[entry.key]);
              if (value === null) return null;
              return { index, value, row };
            })
            .filter(
              (point): point is NonNullable<typeof point> => point !== null,
            );

          if (type === 'line' || type === 'area') {
            const coords = points.map(
              (point) =>
                [bandCenter(point.index), valueAt(point.value)] as const,
            );
            const line = coords
              .map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x} ${y}`)
              .join(' ');

            return (
              <g key={entry.key}>
                {type === 'area' && coords.length > 1 && (
                  <path
                    d={`${line} L ${coords[coords.length - 1][0]} ${zero} L ${coords[0][0]} ${zero} Z`}
                    fill={color}
                    opacity={0.15}
                  />
                )}
                <path
                  d={line}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {points.map((point, i) => (
                  <circle
                    key={point.index}
                    cx={coords[i][0]}
                    cy={coords[i][1]}
                    r={3}
                    fill={color}
                  >
                    <title>{`${categoryLabel(point.row, xKey)} · ${entry.label}: ${fullFormat.format(point.value)}`}</title>
                  </circle>
                ))}
              </g>
            );
          }

          return (
            <g key={entry.key}>
              {points.map((point) => {
                const offset =
                  point.index * band + bandPadding + seriesIndex * barWidth;
                const end = valueAt(point.value);
                const start = Math.min(end, zero);
                const thickness = Math.max(1, Math.abs(end - zero));

                return (
                  <rect
                    key={point.index}
                    x={horizontal ? Math.min(zero, end) : offset}
                    y={horizontal ? offset : start}
                    width={horizontal ? thickness : Math.max(1, barWidth - 1)}
                    height={horizontal ? Math.max(1, barWidth - 1) : thickness}
                    fill={color}
                    rx={2}
                  >
                    <title>{`${categoryLabel(point.row, xKey)} · ${entry.label}: ${fullFormat.format(point.value)}`}</title>
                  </rect>
                );
              })}
            </g>
          );
        })}
      </g>
    </svg>
  );
}

function PieChart({ spec }: { spec: ChartSpec }) {
  const valueKey = spec.series[0];
  const slices = spec.data
    .map((row) => ({
      label: categoryLabel(row, spec.xKey),
      value: toNumber(row[valueKey.key]) ?? 0,
    }))
    .filter((slice) => slice.value > 0);

  const total = slices.reduce((sum, slice) => sum + slice.value, 0);
  if (total === 0) return null;

  const size = 300;
  const radius = size / 2 - 8;
  const center = size / 2;
  let angle = -Math.PI / 2;

  return (
    <div className="flex flex-wrap items-center justify-center gap-6 p-3">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="h-auto w-[220px] shrink-0"
        role="img"
        aria-label={spec.title ?? 'Pie chart'}
      >
        {slices.map((slice, index) => {
          const sweep = (slice.value / total) * Math.PI * 2;
          const start = angle;
          const end = angle + sweep;
          angle = end;
          const color = seriesColor(index);
          const tooltip = `${slice.label}: ${fullFormat.format(slice.value)} (${((slice.value / total) * 100).toFixed(1)}%)`;

          // A lone slice is a full circle, which an arc path cannot express.
          if (slices.length === 1) {
            return (
              <circle
                key={slice.label}
                cx={center}
                cy={center}
                r={radius}
                fill={color}
              >
                <title>{tooltip}</title>
              </circle>
            );
          }

          const x1 = center + radius * Math.cos(start);
          const y1 = center + radius * Math.sin(start);
          const x2 = center + radius * Math.cos(end);
          const y2 = center + radius * Math.sin(end);
          const largeArc = sweep > Math.PI ? 1 : 0;

          return (
            <path
              key={slice.label}
              d={`M ${center} ${center} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z`}
              fill={color}
              stroke="var(--background)"
              strokeWidth={2}
            >
              <title>{tooltip}</title>
            </path>
          );
        })}
      </svg>

      <ul className="flex min-w-0 flex-col gap-1.5 text-sm">
        {slices.map((slice, index) => (
          <li key={slice.label} className="flex items-center gap-2">
            <span
              aria-hidden
              className="size-2.5 shrink-0 rounded-[2px]"
              style={{ backgroundColor: seriesColor(index) }}
            />
            <span className="truncate">{slice.label}</span>
            <span className="ml-auto shrink-0 pl-3 text-muted-foreground tabular-nums">
              {((slice.value / total) * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Renders a chart spec emitted by the agent.
 *
 * The supervisor cannot hand us an image — its hosted code sandbox discards
 * whatever it plots — so charts arrive as data and are drawn here.
 */
export const AgentChart = memo(
  ({ spec, className }: { spec: ChartSpec; className?: string }) => {
    const body = useMemo(
      () =>
        spec.type === 'pie' ? (
          <PieChart spec={spec} />
        ) : (
          <div className="p-3">
            <CartesianChart spec={spec} />
          </div>
        ),
      [spec],
    );

    return (
      <figure
        data-testid="agent-chart"
        className={cn(
          'not-prose w-full overflow-hidden rounded-xl border',
          className,
        )}
      >
        {spec.title && (
          <figcaption className="border-b px-3 py-2 font-medium text-sm">
            {spec.title}
          </figcaption>
        )}
        {body}
        {spec.type !== 'pie' && spec.series.length > 1 && (
          <ChartLegend series={spec.series} />
        )}
      </figure>
    );
  },
);

AgentChart.displayName = 'AgentChart';
