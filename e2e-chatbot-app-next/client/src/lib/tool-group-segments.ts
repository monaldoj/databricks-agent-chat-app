import type { ChatMessage } from '@chat-template/core';

export type ChatPart = ChatMessage['parts'][number];
export type ToolPart = Extract<ChatPart, { type: 'dynamic-tool' }>;

export type RenderBlock =
  | { kind: 'segment'; parts: ChatPart[]; index: number }
  | { kind: 'tool-group'; tools: ToolPart[]; startIndex: number };

export type ToolDisplayGroup = {
  tools: ToolPart[];
  aggregate: boolean;
};

function mcpServerName(tool: ToolPart): string {
  return tool.callProviderMetadata?.databricks?.mcpServerName?.toString() ?? '';
}

/** True only for calls from the configured system.ai.web_search MCP service. */
export function isWebSearchMcpTool(tool: ToolPart): boolean {
  const server = mcpServerName(tool)
    .replace(/[\s_.-]+/g, '')
    .toLowerCase();
  if (server.includes('websearch')) return true;

  // Non-approval MCP results do not always retain their server metadata in
  // older saved chats. Their function name does survive; provider-hosted
  // searches are excluded because the provider executes those directly.
  const toolName = tool.toolName.replace(/-/g, '_').toLowerCase();
  return !tool.providerExecuted && toolName.includes('web_search');
}

/**
 * Coalesce repeated web-search calls into one display group.
 *
 * Other tools remain separate, even when several calls are adjacent. Approval
 * requests also remain separate because each one needs its own action controls.
 */
export function groupRepeatedWebSearchTools(
  tools: ToolPart[],
): ToolDisplayGroup[] {
  const groups: ToolDisplayGroup[] = [];
  const webSearchGroups = new Map<string, ToolDisplayGroup>();

  for (const tool of tools) {
    const canAggregate =
      isWebSearchMcpTool(tool) &&
      tool.callProviderMetadata?.databricks?.approvalRequestId == null;
    if (!canAggregate) {
      groups.push({ tools: [tool], aggregate: false });
      continue;
    }

    const key = `${mcpServerName(tool)}\u0000${tool.toolName}`;
    const existing = webSearchGroups.get(key);
    if (existing) {
      existing.tools.push(tool);
      existing.aggregate = true;
    } else {
      const group = { tools: [tool], aggregate: false };
      webSearchGroups.set(key, group);
      groups.push(group);
    }
  }

  return groups;
}

export function groupConsecutiveToolSegments(
  partSegments: ChatPart[][],
): RenderBlock[] {
  const blocks: RenderBlock[] = [];
  let i = 0;
  while (i < partSegments.length) {
    const segment = partSegments[i];
    const firstPart = segment[0];
    if (firstPart?.type === 'dynamic-tool') {
      const startIndex = i;
      const tools: ToolPart[] = [firstPart as ToolPart];
      i++;
      while (
        i < partSegments.length &&
        partSegments[i][0]?.type === 'dynamic-tool'
      ) {
        tools.push(partSegments[i][0] as ToolPart);
        i++;
      }
      blocks.push({ kind: 'tool-group', tools, startIndex });
    } else {
      blocks.push({ kind: 'segment', parts: segment, index: i });
      i++;
    }
  }
  return blocks;
}
