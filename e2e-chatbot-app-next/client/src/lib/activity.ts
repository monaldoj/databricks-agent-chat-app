import type { ChatMessage } from '@chat-template/core';
import type { ChatPart } from '@/lib/tool-group-segments';
import { parseGenieToolName } from '@/lib/tool-labels';

/** Tool states that mean the agent is still waiting on the tool. */
const RUNNING_TOOL_STATES = new Set([
  'input-streaming',
  'input-available',
  'approval-responded',
]);

/**
 * Turn a tool identifier into something a reader recognises.
 *
 * Genie tools are named after their space id (`query_space_01f169d6…`,
 * `poll_response_01f169d6…`), and built-in tools use double-underscore paths
 * (`system__ai__python_exec`).
 */
export function describeTool(toolName: string): string {
  if (
    parseGenieToolName(toolName) ||
    /^poll_response_/i.test(toolName) ||
    /^genie[-_]/i.test(toolName)
  ) {
    return 'Querying Genie';
  }
  if (toolName === 'web_search' || toolName === 'web_search_preview') {
    return 'Searching the web';
  }
  if (toolName === 'sandbox') return 'Running code';
  if (toolName.endsWith('python_exec')) return 'Running Python';

  const readable = toolName
    .replace(/^system__ai__/, '')
    .replace(/__/g, ' ')
    .replace(/[_-]/g, ' ')
    .trim();
  return `Running ${readable || toolName}`;
}

/**
 * Describe what the agent is doing from the tail of the message it is building.
 * Returns null when the agent is not the one we are waiting on — an approval
 * prompt is waiting on the user, and claiming to be busy would be wrong.
 */
export function describeActivity(
  message: ChatMessage | undefined,
): string | null {
  if (!message || message.role !== 'assistant') return 'Thinking';

  for (let i = message.parts.length - 1; i >= 0; i--) {
    const part: ChatPart = message.parts[i];
    if (part.type === 'dynamic-tool') {
      if (part.state === 'approval-requested') return null;
      return RUNNING_TOOL_STATES.has(part.state)
        ? describeTool(part.toolName)
        : 'Thinking';
    }
    if (part.type === 'text' || part.type === 'reasoning') return 'Thinking';
  }

  return 'Thinking';
}
