import { expect, test } from '@playwright/test';
import {
  parseChartSpec,
  splitChartBlocks,
  toNumber,
} from '../../client/src/lib/chart-spec';

const SAMPLE_SPEC = {
  type: 'bar',
  title: 'Top cities',
  xKey: 'city',
  series: [{ key: 'swipes', label: 'Swipes' }],
  data: [
    { city: 'Toronto', swipes: 421 },
    { city: 'Seattle', swipes: 420 },
  ],
};

test.describe('splitChartBlocks', () => {
  test('leaves plain markdown as a single segment', () => {
    expect(splitChartBlocks('Hello **world**.')).toEqual([
      { kind: 'markdown', text: 'Hello **world**.' },
    ]);
  });

  test('lifts a complete chart block out of surrounding prose', () => {
    const text = `Here is the chart:\n\n\`\`\`chart\n${JSON.stringify(SAMPLE_SPEC)}\n\`\`\`\n\nToronto leads.`;
    const segments = splitChartBlocks(text);

    expect(segments).toHaveLength(3);
    expect(segments[0]).toMatchObject({ kind: 'markdown' });
    expect(segments[1]).toMatchObject({ kind: 'chart', complete: true });
    expect(segments[2]).toMatchObject({
      kind: 'markdown',
      text: '\n\nToronto leads.',
    });
    expect(
      parseChartSpec((segments[1] as { source: string }).source)?.title,
    ).toBe('Top cities');
  });

  test('reports an unclosed chart fence as incomplete instead of leaking JSON', () => {
    const text = 'Working on it:\n```chart\n{"type": "bar", "xKey": "city"';
    const segments = splitChartBlocks(text);

    expect(segments).toHaveLength(2);
    expect(segments[0]).toMatchObject({ kind: 'markdown' });
    expect(segments[1]).toEqual({
      kind: 'chart',
      source: '{"type": "bar", "xKey": "city"',
      complete: false,
    });
  });
});

test.describe('parseChartSpec', () => {
  test('reads a well-formed spec', () => {
    const spec = parseChartSpec(JSON.stringify(SAMPLE_SPEC));
    expect(spec).toMatchObject({
      type: 'bar',
      title: 'Top cities',
      xKey: 'city',
      series: [{ key: 'swipes', label: 'Swipes' }],
    });
    expect(spec?.data).toHaveLength(2);
  });

  test('infers series from numeric fields when series is omitted', () => {
    const spec = parseChartSpec(
      JSON.stringify({
        xKey: 'city',
        data: [
          { city: 'Toronto', swipes: 421 },
          { city: 'Seattle', swipes: 420 },
        ],
      }),
    );
    expect(spec?.series.map((entry) => entry.key)).toEqual(['swipes']);
  });

  test('treats donut as pie and hbar as horizontalBar', () => {
    expect(
      parseChartSpec(JSON.stringify({ ...SAMPLE_SPEC, type: 'donut' }))?.type,
    ).toBe('pie');
    expect(
      parseChartSpec(JSON.stringify({ ...SAMPLE_SPEC, type: 'hbar' }))?.type,
    ).toBe('horizontalBar');
  });

  test('returns null for invalid JSON or empty data', () => {
    expect(parseChartSpec('{')).toBeNull();
    expect(parseChartSpec(JSON.stringify({ data: [] }))).toBeNull();
  });
});

test.describe('toNumber', () => {
  test('tolerates currency and thousands separators', () => {
    expect(toNumber(421)).toBe(421);
    expect(toNumber('1,234.50')).toBe(1234.5);
    expect(toNumber('$1,234.50')).toBe(1234.5);
    expect(toNumber('n/a')).toBeNull();
  });
});
