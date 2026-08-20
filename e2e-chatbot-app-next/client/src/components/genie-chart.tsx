import { memo, useMemo, useState } from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  buildChartSpec,
  toNumber,
  type GenieChartSpec,
  type GenieResultSet,
} from '@/lib/genie-result';
import { cn } from '@/lib/utils';

const CHART_WIDTH = 720;
const VERTICAL_HEIGHT = 340;
const ROW_HEIGHT = 28;

const SERIES_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
];

/**
 * Past the end of the palette hues repeat, so each extra lap is blended toward
 * the background to keep a seventh series distinct from the first.
 */
const seriesColor = (index: number) => {
  const base = SERIES_COLORS[index % SERIES_COLORS.length];
  const lap = Math.floor(index / SERIES_COLORS.length);
  if (lap === 0) return base;
  return `color-mix(in srgb, ${base} ${Math.max(30, 100 - lap * 35)}%, var(--background))`;
};

/** Axis ticks stay short; tooltips carry the exact figure. */
const compactFormat = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  maximumFractionDigits: 1,
});
const fullFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

/** Deliberately generous: an underestimate clips labels off the SVG edge. */
const LABEL_CHAR_WIDTH = 6.6;
const LABEL_MAX_CHARS = 22;
/** Ceiling on how much width category labels may take from the plot. */
const LABEL_WIDTH_RATIO = 0.32;
const ROTATION_DEGREES = 35;
const ROTATION_SIN = Math.sin((ROTATION_DEGREES * Math.PI) / 180);

const truncateLabel = (label: string, maxChars: number) =>
  label.length > maxChars ? `${label.slice(0, maxChars - 1)}…` : label;

type Scale = { min: number; max: number; ticks: number[] };

/** Round a tick so repeated addition doesn't leave floating-point dust. */
const cleanTick = (value: number) => Number(value.toFixed(10));

function niceScale(dataMin: number, dataMax: number, tickCount = 5): Scale {
  // Bars are only honest when measured from zero.
  const min = Math.min(0, dataMin);
  let max = Math.max(0, dataMax);
  if (min === max) max = min + 1;

  const rawStep = (max - min) / tickCount;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step =
    (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;

  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = niceMin; value <= niceMax + step * 1e-9; value += step) {
    ticks.push(cleanTick(value));
  }
  return { min: niceMin, max: niceMax, ticks };
}

function SeriesSwatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="size-2.5 shrink-0 rounded-[2px]"
      style={{ backgroundColor: color }}
    />
  );
}

function ChartLegend({ spec }: { spec: GenieChartSpec }) {
  if (spec.series.length < 2) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 pb-3 text-muted-foreground text-xs">
      {spec.series.map((entry, index) => (
        <span key={entry.column.name} className="flex items-center gap-1.5">
          <SeriesSwatch color={seriesColor(index)} />
          {entry.column.name}
        </span>
      ))}
    </div>
  );
}

/**
 * Tooltip listing every series value for one category.
 *
 * The trigger is an invisible band spanning the plot rather than the mark
 * itself, so thin bars and line points stay easy to hit.
 */
function CategoryTooltip({
  label,
  entries,
  children,
}: {
  label: string;
  entries: { name: string; color: string; value: number | null }[];
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-xs text-xs font-normal"
        collisionPadding={8}
      >
        <div className="mb-1 font-semibold break-words">{label}</div>
        <div className="flex flex-col gap-0.5">
          {entries.map((entry) => (
            <div key={entry.name} className="flex items-center gap-2">
              <SeriesSwatch color={entry.color} />
              <span className="truncate">{entry.name}</span>
              <span className="ml-auto pl-3 tabular-nums font-semibold">
                {entry.value === null ? '—' : fullFormat.format(entry.value)}
              </span>
            </div>
          ))}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

function GenieChartSvg({
  result,
  spec,
}: {
  result: GenieResultSet;
  spec: GenieChartSpec;
}) {
  const [hovered, setHovered] = useState<number | null>(null);

  const horizontal = spec.type === 'horizontalBar';
  const points = spec.rowIndices;

  const labels = points.map((rowIndex) => result.rows[rowIndex][spec.axisIndex] ?? '—');
  const maxLabelChars = Math.min(
    LABEL_MAX_CHARS,
    Math.floor((CHART_WIDTH * LABEL_WIDTH_RATIO) / LABEL_CHAR_WIDTH),
  );
  const shownLabels = labels.map((label) => truncateLabel(label, maxLabelChars));

  const values = points.flatMap((rowIndex) =>
    spec.series
      .map(({ columnIndex }) => toNumber(result.rows[rowIndex][columnIndex]))
      .filter((value): value is number => value !== null),
  );
  const scale = niceScale(Math.min(...values), Math.max(...values));

  const longestLabel = Math.max(...shownLabels.map((label) => label.length));
  const labelWidth = longestLabel * LABEL_CHAR_WIDTH;
  const rotateLabels = !horizontal && longestLabel > 6;

  const chartHeight = horizontal
    ? Math.min(900, Math.max(200, points.length * ROW_HEIGHT + 64))
    : VERTICAL_HEIGHT;

  const margin = horizontal
    ? { top: 8, right: 40, bottom: 40, left: 20 + labelWidth }
    : {
        top: 8,
        right: 16,
        bottom: rotateLabels ? 28 + labelWidth * ROTATION_SIN : 44,
        left: 64,
      };

  const plotWidth = CHART_WIDTH - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;

  const valueAxisLength = horizontal ? plotWidth : plotHeight;
  const valueAt = (value: number) => {
    const ratio = (value - scale.min) / (scale.max - scale.min);
    return horizontal ? ratio * valueAxisLength : valueAxisLength - ratio * valueAxisLength;
  };
  const zero = valueAt(0);

  const band = (horizontal ? plotHeight : plotWidth) / points.length;
  const bandPadding = band * 0.15;
  const barThickness = (band - bandPadding * 2) / spec.series.length;
  const bandStart = (index: number) => index * band;
  const bandCenter = (index: number) => index * band + band / 2;

  // A crowded axis drops labels rather than overlapping them.
  const labelStride = Math.ceil(points.length / (horizontal ? 24 : 14));

  const axisTitle = spec.axisColumn.name;

  return (
    <TooltipProvider delayDuration={0} skipDelayDuration={0}>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${chartHeight}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${spec.series.map((s) => s.column.name).join(', ')} by ${axisTitle}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <g transform={`translate(${margin.left} ${margin.top})`}>
          {scale.ticks.map((tick) => {
            const offset = valueAt(tick);
            return (
              <g key={tick}>
                <line
                  x1={horizontal ? offset : 0}
                  x2={horizontal ? offset : plotWidth}
                  y1={horizontal ? 0 : offset}
                  y2={horizontal ? plotHeight : offset}
                  stroke="currentColor"
                  className="text-border"
                  strokeWidth={1}
                />
                <text
                  x={horizontal ? offset : -10}
                  y={horizontal ? plotHeight + 20 : offset + 4}
                  textAnchor={horizontal ? 'middle' : 'end'}
                  className="fill-muted-foreground text-[11px]"
                >
                  {compactFormat.format(tick)}
                </text>
              </g>
            );
          })}

          {shownLabels.map((label, index) => {
            if (index % labelStride !== 0) return null;
            const center = bandCenter(index);
            const full = labels[index];
            const baseline = plotHeight + 18;

            return (
              <text
                key={`label-${points[index]}`}
                x={horizontal ? -10 : center}
                y={horizontal ? center + 4 : baseline}
                textAnchor={horizontal || rotateLabels ? 'end' : 'middle'}
                transform={
                  rotateLabels
                    ? `rotate(-${ROTATION_DEGREES} ${center} ${baseline})`
                    : undefined
                }
                className={cn(
                  'fill-muted-foreground text-[11px]',
                  hovered === index && 'fill-foreground font-medium',
                )}
              >
                {label}
                {label !== full && <title>{full}</title>}
              </text>
            );
          })}

          <line
            x1={horizontal ? zero : 0}
            x2={horizontal ? zero : plotWidth}
            y1={horizontal ? 0 : zero}
            y2={horizontal ? plotHeight : zero}
            stroke="currentColor"
            className="text-muted-foreground/50"
            strokeWidth={1}
          />

          {spec.series.map((entry, seriesIndex) => {
            const color = seriesColor(seriesIndex);
            const marks = points
              .map((rowIndex, index) => {
                const value = toNumber(result.rows[rowIndex][entry.columnIndex]);
                return value === null ? null : { index, value };
              })
              .filter((mark): mark is { index: number; value: number } => mark !== null);

            if (spec.type === 'line') {
              const coords = marks.map(
                (mark) => [bandCenter(mark.index), valueAt(mark.value)] as const,
              );
              const path = coords
                .map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x} ${y}`)
                .join(' ');

              return (
                <g key={entry.column.name}>
                  <path
                    d={path}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {marks.map((mark, i) => (
                    <circle
                      key={mark.index}
                      cx={coords[i][0]}
                      cy={coords[i][1]}
                      r={hovered === mark.index ? 5 : 3}
                      fill={color}
                    />
                  ))}
                </g>
              );
            }

            return (
              <g key={entry.column.name}>
                {marks.map((mark) => {
                  const offset =
                    bandStart(mark.index) + bandPadding + seriesIndex * barThickness;
                  const end = valueAt(mark.value);
                  const thickness = Math.max(1, Math.abs(end - zero));

                  return (
                    <rect
                      key={mark.index}
                      x={horizontal ? Math.min(zero, end) : offset}
                      y={horizontal ? offset : Math.min(end, zero)}
                      width={horizontal ? thickness : Math.max(1, barThickness - 1)}
                      height={horizontal ? Math.max(1, barThickness - 1) : thickness}
                      fill={color}
                      opacity={hovered === null || hovered === mark.index ? 1 : 0.45}
                      rx={2}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* Invisible hover bands sit on top so every category is easy to hit. */}
          {points.map((rowIndex, index) => (
            <CategoryTooltip
              key={`hit-${rowIndex}`}
              label={labels[index]}
              entries={spec.series.map((entry, seriesIndex) => ({
                name: entry.column.name,
                color: seriesColor(seriesIndex),
                value: toNumber(result.rows[rowIndex][entry.columnIndex]),
              }))}
            >
              <rect
                x={horizontal ? 0 : bandStart(index)}
                y={horizontal ? bandStart(index) : 0}
                width={horizontal ? plotWidth : band}
                height={horizontal ? band : plotHeight}
                className={cn(
                  'cursor-default outline-none',
                  hovered === index ? 'fill-foreground/5' : 'fill-transparent',
                )}
                onPointerEnter={() => setHovered(index)}
                onPointerLeave={() => setHovered((current) => (current === index ? null : current))}
              />
            </CategoryTooltip>
          ))}
        </g>
      </svg>
    </TooltipProvider>
  );
}

function GenieTable({ result }: { result: GenieResultSet }) {
  return (
    <div className="max-h-96 overflow-auto">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="sticky top-0 bg-muted/80 backdrop-blur">
          <tr>
            {result.columns.map((column) => (
              <th
                key={column.name}
                scope="col"
                className={cn(
                  'whitespace-nowrap px-3 py-2 font-medium text-muted-foreground text-xs',
                  column.kind === 'numeric' && 'text-right',
                )}
              >
                {column.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.rows.map((row, rowIndex) => (
            <tr key={`row-${rowIndex}`} className="border-border/60 border-t">
              {result.columns.map((column, columnIndex) => (
                <td
                  key={column.name}
                  className={cn(
                    'px-3 py-1.5',
                    column.kind === 'numeric' && 'text-right tabular-nums',
                  )}
                >
                  {row[columnIndex] ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * One Genie query result, drawn from the rows Genie itself returned.
 *
 * Genie's API carries no chart specification, so a chart is inferred only for
 * time series or rankings too large to scan as a table. The values, their
 * order, and the column names are reproduced exactly as Genie sent them.
 */
export const GenieResultCard = memo(({ result }: { result: GenieResultSet }) => {
  const spec = useMemo(() => buildChartSpec(result), [result]);
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const [showSql, setShowSql] = useState(false);

  const showChart = spec !== null && view === 'chart';
  const heading = spec
    ? `${spec.series.map((entry) => entry.column.name).join(', ')} by ${spec.axisColumn.name}`
    : result.columns.map((column) => column.name).join(', ');

  return (
    <figure
      data-testid="genie-result"
      className="not-prose w-full overflow-hidden rounded-xl border"
    >
      <figcaption className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <span className="min-w-0 flex-1 truncate font-medium text-sm">{heading}</span>
        {spec && (
          <div className="flex shrink-0 items-center gap-1 rounded-md bg-muted/60 p-0.5">
            {(['chart', 'table'] as const).map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setView(option)}
                className={cn(
                  'rounded px-2 py-0.5 text-xs capitalize transition-colors',
                  view === option
                    ? 'bg-background font-medium shadow-sm'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {option}
              </button>
            ))}
          </div>
        )}
      </figcaption>

      {showChart ? (
        <div className="p-3">
          <GenieChartSvg result={result} spec={spec} />
        </div>
      ) : (
        <GenieTable result={result} />
      )}

      {showChart && <ChartLegend spec={spec} />}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t px-3 py-2 text-muted-foreground text-xs">
        <span>
          {result.rows.length} {result.rows.length === 1 ? 'row' : 'rows'}
          {result.truncated && result.totalRowCount
            ? ` of ${fullFormat.format(result.totalRowCount)} (truncated by Genie)`
            : ''}
        </span>
        {result.sql && (
          <button
            type="button"
            onClick={() => setShowSql((current) => !current)}
            className="underline underline-offset-2 hover:text-foreground"
          >
            {showSql ? 'Hide SQL' : 'Show SQL'}
          </button>
        )}
      </div>

      {showSql && result.sql && (
        <pre className="overflow-x-auto border-t bg-muted/40 px-3 py-2 font-mono text-xs">
          {result.sql}
        </pre>
      )}
    </figure>
  );
});

GenieResultCard.displayName = 'GenieResultCard';
