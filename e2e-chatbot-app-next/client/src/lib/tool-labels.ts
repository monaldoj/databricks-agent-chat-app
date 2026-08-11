/**
 * Turns raw tool names into labels a reader can take in at a glance.
 *
 * Genie's MCP server names each tool after the space it belongs to, so a call
 * arrives as `query_space_01f169d6d34a1233bb6d5ab580b58495`. The id is pulled
 * out here so the caller can look up the space's real title and show that.
 */

/** `<action>_<32 hex space id>`, the shape Genie's MCP server generates. */
const GENIE_TOOL_PATTERN = /^([a-z_]+?)_([0-9a-f]{32})$/i;

/**
 * Only these prefixes are treated as Genie tools. An unrecognised action falls
 * through to the generic label rather than being guessed at, so a new Genie
 * tool reads as itself instead of being mislabelled.
 */
const GENIE_ACTION_LABELS: Record<string, string> = {
  query_space: 'Genie',
  get_space_info: 'Genie space details',
};

export type GenieToolName = {
  spaceId: string;
  /** Stands in on its own when the space title cannot be resolved. */
  action: string;
};

export function parseGenieToolName(toolName: string): GenieToolName | null {
  const match = GENIE_TOOL_PATTERN.exec(toolName);
  if (!match) return null;

  const action = GENIE_ACTION_LABELS[match[1].toLowerCase()];
  if (!action) return null;

  return { spaceId: match[2], action };
}

/** `web_search_preview` becomes `Web search preview`. */
export function humanizeToolName(toolName: string): string {
  const spaced = toolName.replace(/[_-]+/g, ' ').trim();
  if (spaced === '') return toolName;
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * The label to show for a tool call. `genieSpaceTitle` is the resolved title
 * for `parseGenieToolName(toolName).spaceId`, or null while it is loading or
 * could not be read.
 */
export function toolDisplayName(
  toolName: string,
  genieSpaceTitle?: string | null,
): string {
  const genie = parseGenieToolName(toolName);
  if (!genie) return humanizeToolName(toolName);
  return genieSpaceTitle ? `${genie.action}: ${genieSpaceTitle}` : genie.action;
}
