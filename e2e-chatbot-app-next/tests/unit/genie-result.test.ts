import { expect, test } from '@playwright/test';
import {
  buildChartSpec,
  parseGenieResults,
  toNumber,
} from '../../client/src/lib/genie-result';

/**
 * Captured from the Genie MCP server (`/api/2.0/mcp/genie/<space_id>`) so the
 * parser is exercised against the wire format rather than an idealised one.
 */
const CITY_ROWS: [string, string][] = [
  ['Toronto', '421'],
  ['Seattle', '420'],
  ['Paris', '419'],
  ['Chicago', '416'],
  ['Boston', '413'],
  ['London', '410'],
  ['New York', '400'],
  ['Austin', '395'],
  ['Tokyo', '384'],
  ['Berlin', '362'],
];

function genieResponse(
  columns: { name: string; type_name: string }[],
  rows: string[][],
  overrides: { statementId?: string; truncated?: boolean; totalRowCount?: number } = {},
) {
  return {
    content: {
      queryAttachments: [
        {
          query: 'SELECT `txn_city`, COUNT(*) AS swipe_count\nFROM `card_swipe_transactions`',
          description: 'You want to see the number of card swipes for each city.',
          statement_response: {
            statement_id: overrides.statementId ?? '01f195a0-a111-1b3b-b259-170b112ad3ee',
            status: { state: 'SUCCEEDED' },
            manifest: {
              format: 'JSON_ARRAY',
              schema: {
                column_count: columns.length,
                columns: columns.map((column, position) => ({
                  ...column,
                  type_text: column.type_name,
                  position,
                })),
              },
              total_row_count: overrides.totalRowCount ?? rows.length,
              truncated: overrides.truncated ?? false,
            },
            result: {
              chunk_index: 0,
              row_offset: 0,
              row_count: rows.length,
              data_array: rows.map((row) => ({
                values: row.map((value) => ({ string_value: value })),
              })),
            },
          },
        },
      ],
      textAttachments: ['Toronto has the highest number of swipes (421).'],
      suggestedQuestions: ['What are the top 10 cities by swipe counts?'],
    },
    conversationId: '01f195a09ed11d6b9f9ff60852e180a3',
    messageId: '01f195a09eda16f8bfbb2047898989fa',
    status: 'COMPLETED',
  };
}

const CITY_COLUMNS = [
  { name: 'txn_city', type_name: 'STRING' },
  { name: 'swipe_count', type_name: 'LONG' },
];

const cityResponse = () => genieResponse(CITY_COLUMNS, CITY_ROWS);

test.describe('parseGenieResults', () => {
  test('reads a Genie response delivered as a JSON string', () => {
    const [result] = parseGenieResults(JSON.stringify(cityResponse()));

    expect(result.columns.map((column) => column.name)).toEqual([
      'txn_city',
      'swipe_count',
    ]);
    expect(result.columns.map((column) => column.kind)).toEqual([
      'categorical',
      'numeric',
    ]);
    expect(result.rows).toHaveLength(10);
    expect(result.rows[0]).toEqual(['Toronto', '421']);
    expect(result.sql).toContain('card_swipe_transactions');
    expect(result.truncated).toBe(false);
  });

  test('reads the same response delivered as an object', () => {
    expect(parseGenieResults(cityResponse())[0].rows).toHaveLength(10);
  });

  test('reads a response wrapped in MCP content blocks', () => {
    const wrapped = [
      { type: 'text', text: JSON.stringify(cityResponse()), annotations: null },
    ];
    expect(parseGenieResults(wrapped)[0].rows[0]).toEqual(['Toronto', '421']);
  });

  test('ignores output from other tools', () => {
    expect(parseGenieResults('a plain web search answer')).toEqual([]);
    expect(parseGenieResults({ results: [{ title: 'A page' }] })).toEqual([]);
  });

  test('flags truncation when Genie returned fewer rows than matched', () => {
    const response = genieResponse(CITY_COLUMNS, CITY_ROWS, { totalRowCount: 500 });
    const [result] = parseGenieResults(response);

    expect(result.truncated).toBe(true);
    expect(result.totalRowCount).toBe(500);
  });
});

test.describe('buildChartSpec', () => {
  test('charts a categorical breakdown as vertical bars', () => {
    const [result] = parseGenieResults(cityResponse());
    const spec = buildChartSpec(result);

    expect(spec?.type).toBe('bar');
    expect(spec?.axisColumn.name).toBe('txn_city');
    expect(spec?.series.map((entry) => entry.column.name)).toEqual(['swipe_count']);
  });

  test("preserves Genie's row order rather than re-sorting", () => {
    const shuffled = [...CITY_ROWS].reverse();
    const [result] = parseGenieResults(genieResponse(CITY_COLUMNS, shuffled));
    const spec = buildChartSpec(result);

    expect(
      spec?.rowIndices.map((index) => result.rows[index][0]),
    ).toEqual(shuffled.map(([city]) => city));
  });

  test('turns sideways once there are too many categories to label', () => {
    const many = Array.from({ length: 15 }, (_, i) => [`city-${i}`, String(100 - i)]);
    const [result] = parseGenieResults(genieResponse(CITY_COLUMNS, many));

    expect(buildChartSpec(result)?.type).toBe('horizontalBar');
  });

  test('draws a line when the axis is a date', () => {
    const columns = [
      { name: 'txn_date', type_name: 'DATE' },
      { name: 'swipe_count', type_name: 'LONG' },
    ];
    const rows = [
      ['2024-01-01', '10'],
      ['2024-01-02', '14'],
      ['2024-01-03', '9'],
    ];
    const [result] = parseGenieResults(genieResponse(columns, rows));
    const spec = buildChartSpec(result);

    expect(spec?.type).toBe('line');
    expect(spec?.axisColumn.name).toBe('txn_date');
  });

  test('groups several measures into one chart', () => {
    const columns = [
      { name: 'txn_city', type_name: 'STRING' },
      { name: 'swipe_count', type_name: 'LONG' },
      { name: 'total_amount', type_name: 'DECIMAL' },
    ];
    const rows = [
      ['Toronto', '421', '10500.25'],
      ['Seattle', '420', '9800.00'],
    ];
    const [result] = parseGenieResults(genieResponse(columns, rows));

    expect(buildChartSpec(result)?.series.map((entry) => entry.column.name)).toEqual([
      'swipe_count',
      'total_amount',
    ]);
  });

  test('leaves a single-row answer as a table', () => {
    const [result] = parseGenieResults(
      genieResponse(CITY_COLUMNS, [['Toronto', '421']]),
    );
    expect(buildChartSpec(result)).toBeNull();
  });

  test('leaves a result with no numeric column as a table', () => {
    const columns = [
      { name: 'txn_city', type_name: 'STRING' },
      { name: 'country', type_name: 'STRING' },
    ];
    const rows = [
      ['Toronto', 'Canada'],
      ['Seattle', 'USA'],
    ];
    const [result] = parseGenieResults(genieResponse(columns, rows));
    expect(buildChartSpec(result)).toBeNull();
  });

  test('leaves a result with repeated axis labels as a table', () => {
    const rows = [
      ['Toronto', '421'],
      ['Toronto', '120'],
      ['Seattle', '420'],
    ];
    const [result] = parseGenieResults(genieResponse(CITY_COLUMNS, rows));
    expect(buildChartSpec(result)).toBeNull();
  });
});

test.describe('toNumber', () => {
  test('accepts the formatted numbers Genie sometimes returns', () => {
    expect(toNumber('1234')).toBe(1234);
    expect(toNumber('$1,234.50')).toBe(1234.5);
    expect(toNumber('12%')).toBe(12);
    expect(toNumber('N/A')).toBeNull();
    expect(toNumber(null)).toBeNull();
  });
});
