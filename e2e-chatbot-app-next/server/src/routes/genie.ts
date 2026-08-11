import {
  Router,
  type Request,
  type Response,
  type Router as RouterType,
} from 'express';
import { getDatabricksToken } from '@chat-template/auth';
import { getHostUrl } from '@chat-template/utils';

export const genieRouter: RouterType = Router();

/**
 * Genie space ids are 32 hex characters. Checking the shape before building a
 * URL keeps a crafted id from reaching anything other than a Genie space.
 */
const SPACE_ID_PATTERN = /^[0-9a-f]{32}$/i;

/** A space title changes about as often as the space is renamed. */
const TITLE_TTL_MS = 60 * 60 * 1000;
/** Retry sooner after a failure so a transient blip is not sticky for an hour. */
const FAILURE_TTL_MS = 5 * 60 * 1000;

const titleCache = new Map<string, { title: string | null; expiresAt: number }>();

async function fetchSpaceTitle(
  spaceId: string,
  token: string,
): Promise<string | null> {
  const host = getHostUrl().replace(/\/$/, '');

  const response = await fetch(`${host}/api/2.0/genie/spaces/${spaceId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    throw new Error(`Genie API returned ${response.status}`);
  }

  const data = (await response.json()) as { title?: unknown };
  return typeof data.title === 'string' && data.title !== '' ? data.title : null;
}

/**
 * Try the signed-in user first, then the app itself.
 *
 * Either identity can be the one that works. The user is who the agent queries
 * Genie as, so they can always read the space — but their forwarded token only
 * carries the scopes the app declares. The app's service principal holds broad
 * scopes but only reaches spaces granted to it in `databricks.yml`.
 */
async function resolveSpaceTitle(
  spaceId: string,
  userToken: string | undefined,
): Promise<string | null> {
  const errors: unknown[] = [];

  if (userToken) {
    try {
      return await fetchSpaceTitle(spaceId, userToken);
    } catch (error) {
      errors.push(error);
    }
  }

  try {
    return await fetchSpaceTitle(spaceId, await getDatabricksToken());
  } catch (error) {
    errors.push(error);
  }

  throw new AggregateError(errors, 'No identity could read the space');
}

/**
 * GET /api/genie/spaces/:spaceId - Resolve a Genie space's display title.
 *
 * Genie's MCP tools are named after the space id, so a transcript otherwise
 * reads "query_space_01f169d6…". Looking the title up here rather than
 * configuring it means the label follows the space when it is renamed.
 *
 * A space we cannot read is not an error the caller can act on — the label
 * just falls back to a generic one — so this answers 200 with a null title.
 */
genieRouter.get('/spaces/:spaceId', async (req: Request, res: Response) => {
  const { spaceId } = req.params;

  if (!SPACE_ID_PATTERN.test(spaceId)) {
    res.status(400).json({ error: 'Not a Genie space id' });
    return;
  }

  const cached = titleCache.get(spaceId);
  if (cached && Date.now() < cached.expiresAt) {
    res.json({ spaceId, title: cached.title });
    return;
  }

  let title: string | null = null;
  let ttl = TITLE_TTL_MS;

  try {
    // Cached across users. Every reader of this app can already reach the
    // spaces it is bound to, and a space title is not sensitive on its own.
    title = await resolveSpaceTitle(
      spaceId,
      req.headers['x-forwarded-access-token'] as string | undefined,
    );
  } catch (error) {
    console.warn(`[genie] Could not resolve title for space ${spaceId}:`, error);
    ttl = FAILURE_TTL_MS;
  }

  titleCache.set(spaceId, { title, expiresAt: Date.now() + ttl });
  res.json({ spaceId, title });
});
