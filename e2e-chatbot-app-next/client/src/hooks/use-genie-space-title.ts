import useSWR from 'swr';

type SpaceTitleResponse = { spaceId: string; title: string | null };

/**
 * The display title of a Genie space, or null while it loads or if it cannot
 * be read. SWR dedupes by URL, so the many tool calls that share a space cost
 * one request between them.
 *
 * Pass undefined to skip the lookup entirely — the caller only has a space id
 * for Genie tools, and hooks cannot be called conditionally.
 */
export function useGenieSpaceTitle(spaceId: string | undefined): string | null {
  const { data } = useSWR<SpaceTitleResponse>(
    spaceId ? `/api/genie/spaces/${spaceId}` : null,
    async (url: string) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Title lookup failed: ${response.status}`);
      return response.json();
    },
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      // A title is stable, and the server caches it too; asking once a session
      // is plenty. A failure leaves the generic label in place rather than
      // retrying behind a label nobody is waiting on.
      revalidateIfStale: false,
      shouldRetryOnError: false,
    },
  );

  return data?.title ?? null;
}
