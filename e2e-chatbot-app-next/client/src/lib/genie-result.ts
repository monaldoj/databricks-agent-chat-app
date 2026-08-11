/**
 * Reads the query results a Genie space returns through its MCP server.
 *
 * Genie answers with the SQL it ran, the column schema, and the result rows —
 * it does not return a chart specification, so the rows here are the closest
 * thing to "what Genie drew". Everything downstream renders these values
 * verbatim; nothing is re-derived from the model's prose.
 */

export type GenieColumnKind = 'numeric' | 'temporal' | 'categorical';

export type GenieColumn = {
  name: string;
  typeName: string;
  kind: GenieColumnKind;
};

export type GenieResultSet = {
  /** Stable across re-renders so React keys survive streaming updates. */
  id: string;
  sql?: string;
  description?: string;
  columns: GenieColumn[];
  rows: (string | null)[][];
  /** Rows the query matched, which exceeds `rows.length` when Genie truncated. */
  totalRowCount?: number;
  truncated: boolean;
};

const NUMERIC_TYPES = new Set([
  'BYTE',
  'SHORT',
  'INT',
  'INTEGER',
  'LONG',
  'BIGINT',
  'FLOAT',
  'REAL',
  'DOUBLE',
  'DECIMAL',
  'NUMERIC',
]);

const TEMPORAL_TYPES = new Set([
  'DATE',
  'TIMESTAMP',
  'TIMESTAMP_NTZ',
  'TIMESTAMP_LTZ',
]);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Hunt for Genie's `queryAttachments` anywhere inside a tool output.
 *
 * The payload reaches us wrapped a different number of times depending on how
 * the agent framework serialised the MCP result — sometimes a JSON string,
 * sometimes a list of content blocks whose `text` is itself JSON. Searching
 * for the key is steadier than guessing the nesting.
 */
function findQueryAttachments(value: unknown, depth = 0): unknown[] | null {
  if (depth > 8) return null;

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return null;
    try {
      return findQueryAttachments(JSON.parse(trimmed), depth + 1);
    } catch {
      return null;
    }
  }

  if (Array.isArray(value)) {
    for (const entry of value) {
      const found = findQueryAttachments(entry, depth + 1);
      if (found) return found;
    }
    return null;
  }

  if (isRecord(value)) {
    if (Array.isArray(value.queryAttachments)) return value.queryAttachments;
    for (const entry of Object.values(value)) {
      const found = findQueryAttachments(entry, depth + 1);
      if (found) return found;
    }
  }

  return null;
}

/** Cells arrive either as plain scalars or as protobuf-style value wrappers. */
function readCell(cell: unknown): string | null {
  if (cell === null || cell === undefined) return null;
  if (typeof cell === 'string') return cell;
  if (typeof cell === 'number' || typeof cell === 'boolean') return String(cell);

  if (isRecord(cell)) {
    if (cell.null_value !== undefined) return null;
    for (const key of [
      'string_value',
      'number_value',
      'double_value',
      'long_value',
      'int_value',
      'bool_value',
      'value',
    ]) {
      const inner = cell[key];
      if (inner !== undefined && inner !== null) return String(inner);
    }
  }

  return null;
}

function readRow(row: unknown): (string | null)[] | null {
  if (Array.isArray(row)) return row.map(readCell);
  if (isRecord(row) && Array.isArray(row.values)) return row.values.map(readCell);
  return null;
}

function classifyColumn(typeName: string): GenieColumnKind {
  const normalized = typeName.toUpperCase();
  if (NUMERIC_TYPES.has(normalized)) return 'numeric';
  if (TEMPORAL_TYPES.has(normalized)) return 'temporal';
  // DECIMAL(10,2) and friends carry precision in the type text.
  if (/^(DECIMAL|NUMERIC)\s*\(/.test(normalized)) return 'numeric';
  return 'categorical';
}

function readColumns(manifest: unknown): GenieColumn[] {
  if (!isRecord(manifest)) return [];
  const schema = manifest.schema;
  if (!isRecord(schema) || !Array.isArray(schema.columns)) return [];

  return schema.columns.flatMap((column): GenieColumn[] => {
    if (!isRecord(column) || typeof column.name !== 'string') return [];
    const typeName =
      typeof column.type_name === 'string'
        ? column.type_name
        : typeof column.type_text === 'string'
          ? column.type_text
          : 'STRING';
    return [{ name: column.name, typeName, kind: classifyColumn(typeName) }];
  });
}

function readResultSet(
  attachment: unknown,
  fallbackId: string,
): GenieResultSet | null {
  if (!isRecord(attachment)) return null;

  const statement = attachment.statement_response;
  if (!isRecord(statement)) return null;

  const columns = readColumns(statement.manifest);
  if (columns.length === 0) return null;

  const result = isRecord(statement.result) ? statement.result : null;
  const dataArray = result && Array.isArray(result.data_array) ? result.data_array : [];
  const rows = dataArray
    .map(readRow)
    .filter((row): row is (string | null)[] => row !== null);
  if (rows.length === 0) return null;

  const manifest = isRecord(statement.manifest) ? statement.manifest : null;
  const totalRowCount =
    manifest && typeof manifest.total_row_count === 'number'
      ? manifest.total_row_count
      : undefined;

  return {
    id:
      typeof statement.statement_id === 'string'
        ? statement.statement_id
        : fallbackId,
    sql: typeof attachment.query === 'string' ? attachment.query : undefined,
    description:
      typeof attachment.description === 'string' ? attachment.description : undefined,
    columns,
    rows,
    totalRowCount,
    truncated:
      (manifest && manifest.truncated === true) ||
      (totalRowCount !== undefined && totalRowCount > rows.length),
  };
}

/** Pull every Genie result set out of one tool output, or an empty list. */
export function parseGenieResults(output: unknown): GenieResultSet[] {
  const attachments = findQueryAttachments(output);
  if (!attachments) return [];

  const seen = new Set<string>();
  return attachments.flatMap((attachment, index) => {
    const resultSet = readResultSet(attachment, `attachment-${index}`);
    if (!resultSet || seen.has(resultSet.id)) return [];
    seen.add(resultSet.id);
    return [resultSet];
  });
}

/** Coerce a Genie cell to a number, tolerating "1,234" and "$1,234.50". */
export function toNumber(value: string | null): number | null {
  if (value === null) return null;
  const cleaned = value.replace(/[$,\s%]/g, '');
  if (cleaned === '') return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

export type GenieChartType = 'bar' | 'horizontalBar' | 'line';

export type GenieSeries = {
  column: GenieColumn;
  columnIndex: number;
};

export type GenieChartSpec = {
  type: GenieChartType;
  axisColumn: GenieColumn;
  axisIndex: number;
  series: GenieSeries[];
  /** Row indices into `GenieResultSet.rows`, in Genie's own ordering. */
  rowIndices: number[];
};

/** Beyond this a vertical bar chart stops being legible, so bars go sideways. */
const HORIZONTAL_THRESHOLD = 12;
/** The palette repeats past this, and the legend stops being readable. */
const MAX_SERIES = 6;

/**
 * Decide how to draw a result set, or return null to leave it as a table.
 *
 * Genie's API exposes no chart type, so the shape is inferred from the column
 * schema alone. Row order is never changed — Genie's ORDER BY is the intended
 * ordering and re-sorting here would misrepresent the answer.
 */
export function buildChartSpec(result: GenieResultSet): GenieChartSpec | null {
  // One row is a single fact, better read as a table than plotted.
  if (result.rows.length < 2) return null;

  const series = result.columns
    .map((column, columnIndex) => ({ column, columnIndex }))
    .filter(({ column }) => column.kind === 'numeric')
    .slice(0, MAX_SERIES);
  if (series.length === 0) return null;

  const temporalIndex = result.columns.findIndex((c) => c.kind === 'temporal');
  const categoricalIndex = result.columns.findIndex((c) => c.kind === 'categorical');
  const axisIndex = temporalIndex >= 0 ? temporalIndex : categoricalIndex;
  if (axisIndex < 0) return null;

  // A category axis is only meaningful when its labels are distinct.
  const labels = result.rows.map((row) => row[axisIndex]);
  if (new Set(labels).size < labels.length) return null;

  // Rows where no series has a usable number would render as silent gaps.
  const rowIndices = result.rows
    .map((_, index) => index)
    .filter((index) =>
      series.some(({ columnIndex }) => toNumber(result.rows[index][columnIndex]) !== null),
    );
  if (rowIndices.length < 2) return null;

  const type: GenieChartType =
    temporalIndex >= 0
      ? 'line'
      : rowIndices.length > HORIZONTAL_THRESHOLD
        ? 'horizontalBar'
        : 'bar';

  return {
    type,
    axisColumn: result.columns[axisIndex],
    axisIndex,
    series,
    rowIndices,
  };
}
