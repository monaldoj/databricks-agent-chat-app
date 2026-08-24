import { expect, test } from '@playwright/test';
import type { ChatMessage } from '@chat-template/core';
import {
  collectSourceUrlParts,
  sourceLinkLabel,
} from '../../client/src/components/databricks-message-citation';
import { createMessagePartSegments } from '../../client/src/components/databricks-message-part-transformers';

test.describe('sourceLinkLabel', () => {
  test('uses a page title when one is provided', () => {
    expect(
      sourceLinkLabel(
        'Toronto - Wikipedia',
        'https://en.wikipedia.org/wiki/Toronto',
      ),
    ).toBe('Toronto - Wikipedia');
  });

  test('falls back to the hostname for a raw URL', () => {
    expect(
      sourceLinkLabel(
        'https://www.thestar.com/news',
        'https://www.thestar.com/news',
      ),
    ).toBe('thestar.com');
    expect(sourceLinkLabel(undefined, 'https://www.bbc.com/news')).toBe(
      'bbc.com',
    );
  });

  test('returns the original string when the href is not a URL', () => {
    expect(sourceLinkLabel('Notes', 'not-a-url')).toBe('Notes');
  });
});

const source = (
  url: string,
  title?: string,
): Extract<ChatMessage['parts'][number], { type: 'source-url' }> => ({
  type: 'source-url',
  sourceId: url,
  url,
  title,
});

test.describe('collectSourceUrlParts', () => {
  test('keeps unique sources in first-seen order', () => {
    expect(
      collectSourceUrlParts([
        { type: 'text', text: 'Hello' },
        source('https://cms.gov', 'CMS'),
        { type: 'text', text: 'More' },
        source('https://medicaid.gov', 'Medicaid'),
        source('https://cms.gov', 'Centers for Medicare'),
      ]).map((part) => part.url),
    ).toEqual(['https://cms.gov', 'https://medicaid.gov']);
  });
});

test.describe('createMessagePartSegments', () => {
  test('omits source-url parts so they are not rendered mid-answer', () => {
    const segments = createMessagePartSegments([
      source('https://cms.gov', 'CMS'),
      { type: 'text', text: 'After search. ' },
      { type: 'text', text: 'The rest of the answer.' },
    ]);

    expect(segments).toHaveLength(1);
    expect(segments[0].map((part) => part.type)).toEqual(['text', 'text']);
  });
});
