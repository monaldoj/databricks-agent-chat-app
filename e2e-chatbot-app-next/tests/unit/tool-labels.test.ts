import { expect, test } from '@playwright/test';
import {
  humanizeToolName,
  parseGenieToolName,
  toolDisplayName,
} from '../../client/src/lib/tool-labels';

/** The name the Genie MCP server actually emitted for the bundled space. */
const GENIE_QUERY_TOOL = 'query_space_01f169d6d34a1233bb6d5ab580b58495';
const SPACE_ID = '01f169d6d34a1233bb6d5ab580b58495';

test('pulls the space id out of a Genie query tool', () => {
  expect(parseGenieToolName(GENIE_QUERY_TOOL)).toEqual({
    spaceId: SPACE_ID,
    action: 'Genie',
  });
});

test('recognises a multi-word Genie action prefix', () => {
  expect(parseGenieToolName(`get_space_info_${SPACE_ID}`)).toEqual({
    spaceId: SPACE_ID,
    action: 'Genie space details',
  });
});

test('leaves non-Genie tools alone', () => {
  expect(parseGenieToolName('web_search')).toBeNull();
  expect(parseGenieToolName('python_exec')).toBeNull();
});

test('does not claim an unknown action that merely ends in an id', () => {
  expect(parseGenieToolName(`delete_everything_${SPACE_ID}`)).toBeNull();
});

test('ignores a trailing token that is not a 32 character id', () => {
  expect(parseGenieToolName('query_space_01f169d6')).toBeNull();
});

test('shows the space title once it resolves', () => {
  expect(toolDisplayName(GENIE_QUERY_TOOL, 'Campus Card Swipe Analytics')).toBe(
    'Genie: Campus Card Swipe Analytics',
  );
});

test('falls back to a generic label, never the raw id', () => {
  for (const title of [null, undefined, '']) {
    const label = toolDisplayName(GENIE_QUERY_TOOL, title);
    expect(label).toBe('Genie');
    expect(label).not.toContain(SPACE_ID);
  }
});

test('humanizes other tool names', () => {
  expect(humanizeToolName('web_search_preview')).toBe('Web search preview');
  expect(toolDisplayName('web_search')).toBe('Web search');
});

test('leaves an already readable name readable', () => {
  expect(humanizeToolName('Web search')).toBe('Web search');
  expect(humanizeToolName('_')).toBe('_');
});
