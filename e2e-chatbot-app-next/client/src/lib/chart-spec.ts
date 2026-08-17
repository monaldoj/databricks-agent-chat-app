export type ChartType = 'bar' | 'horizontalBar' | 'line' | 'area' | 'pie';

export type ChartSeries = {
  key: string;
  label: string;
};

export type ChartSpec = {
  type: ChartType;
  title?: string;
  xKey: string;
  xLabel?: string;
  yLabel?: string;
  series: ChartSeries[];
  data: Record<string, unknown>[];
};

export type MarkdownSegment =
  | { kind: 'markdown'; text: string }
  | { kind: 'chart'; source: string; complete: boolean };

const CHART_TYPES: ChartType[] = [
  'bar',
  'horizontalBar',
  'line',
  'area',
  'pie',
];

/** Opening fence of a ```chart block, at the start of a line. */
const CHART_FENCE_OPEN = /(^|\n)[ \t]*```[ \t]*chart[ \t]*(?:\r?\n|$)/g;
/** Closing fence, which must sit on its own line. */
const CHART_FENCE_CLOSE = /\n[ \t]*```[ \t]*(?=\r?\n|$)/;

/**
 * Split assistant markdown into plain-markdown runs and ```chart blocks.
 *
 * A block with no closing fence yet is reported as incomplete rather than
 * dropped, so a chart streaming in can show a placeholder instead of leaking
 * half a JSON object into the transcript.
 */
export function splitChartBlocks(text: string): MarkdownSegment[] {
  const segments: MarkdownSegment[] = [];
  let cursor = 0;
  CHART_FENCE_OPEN.lastIndex = 0;

  let open = CHART_FENCE_OPEN.exec(text);
  while (open !== null) {
    const fenceStart = open.index + open[1].length;
    const bodyStart = open.index + open[0].length;

    if (fenceStart > cursor) {
      segments.push({ kind: 'markdown', text: text.slice(cursor, fenceStart) });
    }

    const tail = text.slice(bodyStart);
    const close = CHART_FENCE_CLOSE.exec(tail);

    if (!close) {
      segments.push({ kind: 'chart', source: tail, complete: false });
      return segments;
    }

    segments.push({
      kind: 'chart',
      source: tail.slice(0, close.index),
      complete: true,
    });
    cursor = bodyStart + close.index + close[0].length;
    CHART_FENCE_OPEN.lastIndex = cursor;
    open = CHART_FENCE_OPEN.exec(text);
  }

  if (cursor < text.length) {
    segments.push({ kind: 'markdown', text: text.slice(cursor) });
  }

  return segments;
}

/** Coerce a cell to a number, tolerating "1,234" and "$1,234.50". */
export function toNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;
  const cleaned = value.replace(/[$,\s%]/g, '');
  if (cleaned === '') return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeType(value: unknown): ChartType {
  if (typeof value !== 'string') return 'bar';
  const match = CHART_TYPES.find(
    (type) => type.toLowerCase() === value.toLowerCase().replace(/[\s_-]/g, ''),
  );
  if (match) return match;
  if (/^(donut|doughnut)$/i.test(value)) return 'pie';
  if (/^(hbar|barh)$/i.test(value)) return 'horizontalBar';
  return 'bar';
}

/**
 * Fall back to every field that holds a number in the first row. Lets a spec
 * that names only `xKey` still plot something useful.
 */
function inferSeries(
  data: Record<string, unknown>[],
  xKey: string,
): ChartSeries[] {
  const first = data[0] ?? {};
  return Object.keys(first)
    .filter((key) => key !== xKey && toNumber(first[key]) !== null)
    .map((key) => ({ key, label: key }));
}

/**
 * Parse the JSON inside a ```chart block. Returns null for anything we cannot
 * plot, and the caller falls back to showing the raw block.
 */
export function parseChartSpec(source: string): ChartSpec | null {
  let raw: unknown;
  try {
    raw = JSON.parse(source);
  } catch {
    return null;
  }

  if (typeof raw !== 'object' || raw === null) return null;
  const spec = raw as Record<string, unknown>;

  const data = Array.isArray(spec.data)
    ? spec.data.filter(
        (row): row is Record<string, unknown> =>
          typeof row === 'object' && row !== null,
      )
    : [];
  if (data.length === 0) return null;

  const xKey =
    typeof spec.xKey === 'string' && spec.xKey in (data[0] ?? {})
      ? spec.xKey
      : Object.keys(data[0] ?? {}).find(
          (key) => toNumber(data[0][key]) === null,
        );
  if (!xKey) return null;

  const declared = Array.isArray(spec.series)
    ? spec.series
        .map((entry) => {
          if (typeof entry === 'string') return { key: entry, label: entry };
          if (typeof entry !== 'object' || entry === null) return null;
          const item = entry as Record<string, unknown>;
          if (typeof item.key !== 'string') return null;
          return {
            key: item.key,
            label: typeof item.label === 'string' ? item.label : item.key,
          };
        })
        .filter((entry): entry is ChartSeries => entry !== null)
        .filter((entry) => entry.key !== xKey)
    : [];

  const series = declared.length > 0 ? declared : inferSeries(data, xKey);
  if (series.length === 0) return null;

  // A row where no series has a usable number would render as a gap.
  const usable = data.filter((row) =>
    series.some((entry) => toNumber(row[entry.key]) !== null),
  );
  if (usable.length === 0) return null;

  return {
    type: normalizeType(spec.type),
    title: typeof spec.title === 'string' ? spec.title : undefined,
    xKey,
    xLabel: typeof spec.xLabel === 'string' ? spec.xLabel : undefined,
    yLabel: typeof spec.yLabel === 'string' ? spec.yLabel : undefined,
    series,
    data: usable,
  };
}
