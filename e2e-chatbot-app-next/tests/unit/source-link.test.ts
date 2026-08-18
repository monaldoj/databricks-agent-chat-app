import { expect, test } from '@playwright/test';
import { sourceLinkLabel } from '../../client/src/components/databricks-message-citation';

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
