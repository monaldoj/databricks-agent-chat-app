import React, { memo, useState } from 'react';
import { MessageMarkdown } from './message-markdown';
import { MessageContent } from './elements/message';
import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
  type ToolState,
} from './elements/tool';
import {
  McpTool,
  McpToolHeader,
  McpToolContent,
  McpToolInput,
  McpApprovalActions,
} from './elements/mcp-tool';
import { MessageActions } from './message-actions';
import { PreviewAttachment } from './preview-attachment';
import equal from 'fast-deep-equal';
import { cn, sanitizeText } from '@/lib/utils';
import { MessageEditor } from './message-editor';
import { MessageReasoning } from './message-reasoning';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { ChatMessage, Feedback } from '@chat-template/core';
import { useDataStream } from './data-stream-provider';
import {
  createMessagePartSegments,
  formatNamePart,
  isNamePart,
  joinMessagePartSegments,
} from './databricks-message-part-transformers';
import {
  collectSourceUrlParts,
  MessageSources,
} from './databricks-message-citation';
import { MessageError } from './message-error';
import { MessageOAuthError } from './message-oauth-error';
import { isCredentialErrorMessage } from '@/lib/oauth-error-utils';
import {
  groupConsecutiveToolSegments,
  groupRepeatedWebSearchTools,
  type ToolPart,
} from '@/lib/tool-group-segments';
import { Streamdown } from 'streamdown';
import { useApproval } from '@/hooks/use-approval';
import { GenieResultCard } from './genie-chart';
import { parseGenieResults, type GenieResultSet } from '@/lib/genie-result';
import { parseGenieToolName, toolDisplayName } from '@/lib/tool-labels';
import { useGenieSpaceTitle } from '@/hooks/use-genie-space-title';
import { ActivityIndicator } from './activity-indicator';

const PurePreviewMessage = ({
  message,
  allMessages,
  isLoading,
  setMessages,
  addToolApprovalResponse,
  sendMessage,
  regenerate,
  isReadonly,
  initialFeedback,
  activityStatus,
}: {
  message: ChatMessage;
  allMessages: ChatMessage[];
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  addToolApprovalResponse: UseChatHelpers<ChatMessage>['addToolApprovalResponse'];
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  regenerate: UseChatHelpers<ChatMessage>['regenerate'];
  isReadonly: boolean;
  initialFeedback?: Feedback;
  activityStatus?: UseChatHelpers<ChatMessage>['status'];
}) => {
  const [mode, setMode] = useState<'view' | 'edit'>('view');
  const [showErrors, setShowErrors] = useState(false);

  // Hook for handling MCP approval requests
  const { submitApproval, isSubmitting, pendingApprovalId } = useApproval({
    addToolApprovalResponse,
    sendMessage,
  });

  const attachmentsFromMessage = message.parts.filter(
    (part) => part.type === 'file',
  );

  // Extract non-OAuth error parts separately (OAuth errors are rendered inline)
  const errorParts = React.useMemo(
    () =>
      message.parts
        .filter((part) => part.type === 'data-error')
        .filter((part) => {
          // OAuth errors are rendered inline, not in the error section
          return !isCredentialErrorMessage(part.data);
        }),
    [message.parts],
  );

  useDataStream();

  const sourceParts = React.useMemo(
    () => collectSourceUrlParts(message.parts),
    [message.parts],
  );

  const partSegments = React.useMemo(
    /**
     * We segment message parts into segments that can be rendered as a single component.
     * Note: OAuth errors are included here for inline rendering, non-OAuth errors are filtered out.
     * Source citations are collected separately and rendered at the bottom.
     */
    () =>
      createMessagePartSegments(
        message.parts.filter(
          (part) =>
            part.type !== 'data-error' || isCredentialErrorMessage(part.data),
        ),
      ),
    [message.parts],
  );

  const renderBlocks = React.useMemo(
    () => groupConsecutiveToolSegments(partSegments),
    [partSegments],
  );

  // Check if message only contains non-OAuth errors (no other content)
  const hasOnlyErrors = React.useMemo(() => {
    const nonErrorParts = message.parts.filter(
      (part) => part.type !== 'data-error',
    );
    // Only consider non-OAuth errors for this check
    return errorParts.length > 0 && nonErrorParts.length === 0;
  }, [message.parts, errorParts.length]);

  return (
    <div
      data-testid={`message-${message.role}`}
      className="group/message w-full"
      data-role={message.role}
    >
      <div
        className={cn('flex w-full items-start gap-2 md:gap-3', {
          'justify-end': message.role === 'user',
          'justify-start': message.role === 'assistant',
        })}
      >
        <div
          className={cn('flex min-w-0 flex-col gap-3', {
            'w-full': message.role === 'assistant' || mode === 'edit',
            'max-w-[70%] sm:max-w-[min(fit-content,80%)]':
              message.role === 'user' && mode !== 'edit',
          })}
        >
          {attachmentsFromMessage.length > 0 && (
            <div
              data-testid={`message-attachments`}
              className={cn('flex flex-row justify-end gap-2', {
                'justify-start': message.role === 'assistant',
              })}
            >
              {attachmentsFromMessage.map((attachment) => (
                <PreviewAttachment
                  key={attachment.url}
                  attachment={{
                    name: attachment.filename ?? 'file',
                    contentType: attachment.mediaType,
                    url: attachment.url,
                  }}
                />
              ))}
            </div>
          )}

          {renderBlocks.map((block) => {
            if (block.kind === 'tool-group') {
              return (
                <React.Fragment key={`tool-group-${block.startIndex}`}>
                  <MessageToolGroup
                    tools={block.tools}
                    isLoading={isLoading}
                    submitApproval={submitApproval}
                    isSubmitting={isSubmitting}
                    pendingApprovalId={pendingApprovalId}
                  />
                  <GenieToolResults tools={block.tools} />
                </React.Fragment>
              );
            }

            const parts = block.parts;
            const index = block.index;
            const [part] = parts;
            const { type } = part;
            const key = `message-${message.id}-part-${index}`;

            if (type === 'reasoning' && part.text?.trim().length > 0) {
              return (
                <MessageReasoning
                  key={key}
                  isLoading={isLoading}
                  reasoning={part.text}
                />
              );
            }

            if (type === 'text') {
              if (isNamePart(part)) {
                return (
                  <Streamdown
                    key={key}
                    className="-mb-2 mt-0 border-l-4 pl-2 text-muted-foreground"
                  >{`# ${formatNamePart(part)}`}</Streamdown>
                );
              }
              if (mode === 'view') {
                return (
                  <div key={key}>
                    <MessageContent
                      data-testid="message-content"
                      className={cn({
                        'w-fit break-words rounded-2xl bg-secondary px-3 py-2 text-left text-base':
                          message.role === 'user',
                        'bg-transparent px-0 py-0 text-left text-base':
                          message.role === 'assistant',
                      })}
                    >
                      <MessageMarkdown>
                        {sanitizeText(joinMessagePartSegments(parts))}
                      </MessageMarkdown>
                    </MessageContent>
                  </div>
                );
              }

              if (mode === 'edit') {
                return (
                  <div
                    key={key}
                    className="flex w-full flex-row items-start gap-3"
                  >
                    <div className="size-8" />
                    <div className="min-w-0 flex-1">
                      <MessageEditor
                        key={message.id}
                        message={message}
                        setMode={setMode}
                        setMessages={setMessages}
                        regenerate={regenerate}
                      />
                    </div>
                  </div>
                );
              }
            }

            // dynamic-tool parts are rendered by MessageToolGroup above.
            // source-url parts are collected into MessageSources below.

            // Render OAuth errors inline
            if (type === 'data-error' && isCredentialErrorMessage(part.data)) {
              return (
                <MessageOAuthError
                  key={key}
                  error={part.data}
                  allMessages={allMessages}
                  setMessages={setMessages}
                  sendMessage={sendMessage}
                />
              );
            }
          })}

          {activityStatus && (
            <ActivityIndicator
              embedded
              status={activityStatus}
              lastMessage={message}
            />
          )}

          {message.role === 'assistant' && (
            <MessageSources sources={sourceParts} />
          )}

          {!isReadonly && !hasOnlyErrors && (
            <MessageActions
              key={`action-${message.id}`}
              message={message}
              isLoading={isLoading}
              setMode={setMode}
              errorCount={errorParts.length}
              showErrors={showErrors}
              onToggleErrors={() => setShowErrors(!showErrors)}
              initialFeedback={initialFeedback}
            />
          )}

          {errorParts.length > 0 && (hasOnlyErrors || showErrors) && (
            <div className="flex flex-col gap-2">
              {errorParts.map((part, index) => (
                <MessageError
                  key={`error-${message.id}-${index}`}
                  error={part.data}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const PreviewMessage = memo(
  PurePreviewMessage,
  (prevProps, nextProps) => {
    if (prevProps.isLoading !== nextProps.isLoading) return false;
    if (prevProps.activityStatus !== nextProps.activityStatus) return false;
    // Always re-render the in-progress assistant message. Deep-equal on parts
    // short-circuits when the SDK mutates the same array, which would freeze
    // the transcript until the turn ended.
    if (nextProps.isLoading) return false;

    if (prevProps.message.id !== nextProps.message.id) return false;
    if (!equal(prevProps.message.parts, nextProps.message.parts)) return false;
    if (
      prevProps.initialFeedback?.feedbackType !==
      nextProps.initialFeedback?.feedbackType
    )
      return false;

    return true;
  },
);

const MessageToolGroup = ({
  tools,
  isLoading,
  submitApproval,
  isSubmitting,
  pendingApprovalId,
}: {
  tools: ToolPart[];
  isLoading: boolean;
  submitApproval: ReturnType<typeof useApproval>['submitApproval'];
  isSubmitting: boolean;
  pendingApprovalId: string | null;
}) => {
  const displayGroups = groupRepeatedWebSearchTools(tools);
  return (
    <div className="flex flex-col gap-2" data-testid="tool-group">
      {displayGroups.map((group) =>
        group.aggregate ? (
          <WebSearchToolGroup
            key={`web-search-${group.tools[0].toolCallId}`}
            tools={group.tools}
            isLoading={isLoading}
          />
        ) : (
          <ToolPartRenderer
            key={group.tools[0].toolCallId}
            part={group.tools[0]}
            isLoading={isLoading}
            submitApproval={submitApproval}
            isSubmitting={isSubmitting}
            pendingApprovalId={pendingApprovalId}
          />
        ),
      )}
    </div>
  );
};

function effectiveToolState(part: ToolPart, isLoading: boolean): ToolState {
  if (part.providerExecuted && !isLoading && part.state === 'input-available') {
    return 'output-available';
  }
  return part.state;
}

function combinedToolState(tools: ToolPart[], isLoading: boolean): ToolState {
  const states = tools.map((tool) => effectiveToolState(tool, isLoading));
  if (states.includes('output-error')) return 'output-error';
  if (states.includes('input-streaming')) return 'input-streaming';
  if (states.includes('input-available')) return 'input-available';
  if (states.includes('approval-responded')) return 'approval-responded';
  if (states.includes('output-denied')) return 'output-denied';
  return 'output-available';
}

const WebSearchToolGroup = ({
  tools,
  isLoading,
}: {
  tools: ToolPart[];
  isLoading: boolean;
}) => (
  <Tool defaultOpen={false}>
    <ToolHeader
      type={`Web search · ${tools.length} searches`}
      state={combinedToolState(tools, isLoading)}
    />
    <ToolContent>
      {tools.map((tool, index) => (
        <div
          className="border-border/60 border-t first:border-t-0"
          key={tool.toolCallId}
        >
          <div className="px-3 pt-3 font-medium text-muted-foreground text-xs">
            Search {index + 1}
          </div>
          <ToolInput input={tool.input} />
          {(tool.output != null || tool.errorText) && (
            <ToolOutput
              output={
                tool.errorText ? undefined : (
                  <div className="whitespace-pre-wrap font-mono text-sm">
                    {typeof tool.output === 'string'
                      ? tool.output
                      : JSON.stringify(tool.output, null, 2)}
                  </div>
                )
              }
              errorText={tool.errorText}
            />
          )}
        </div>
      ))}
    </ToolContent>
  </Tool>
);

/**
 * Renders the query results a Genie space returned, alongside its tool call.
 *
 * Genie sends back its schema and rows in the tool output, so the chart is
 * drawn from that payload directly and never from the model's summary of it.
 */
const GenieToolResults = ({ tools }: { tools: ToolPart[] }) => {
  const results = React.useMemo(() => {
    const seen = new Set<string>();
    const collected: GenieResultSet[] = [];

    for (const tool of tools) {
      if (tool.state !== 'output-available') continue;
      for (const result of parseGenieResults(tool.output)) {
        // A follow-up poll repeats the results of the query that started it.
        if (seen.has(result.id)) continue;
        seen.add(result.id);
        collected.push(result);
      }
    }

    return collected;
  }, [tools]);

  if (results.length === 0) return null;

  return (
    <div className="flex flex-col gap-3">
      {results.map((result) => (
        <GenieResultCard key={result.id} result={result} />
      ))}
    </div>
  );
};

const ToolPartRenderer = ({
  part,
  isLoading,
  submitApproval,
  isSubmitting,
  pendingApprovalId,
}: {
  part: ToolPart;
  isLoading: boolean;
  submitApproval: ReturnType<typeof useApproval>['submitApproval'];
  isSubmitting: boolean;
  pendingApprovalId: string | null;
}) => {
  const { toolCallId, input, state, errorText, output, toolName } = part;

  // Genie names its tools after the space id, which says nothing to a reader,
  // so the space's own title stands in for it where one can be resolved.
  const genieSpaceId = parseGenieToolName(toolName)?.spaceId;
  const displayName = toolDisplayName(
    toolName,
    useGenieSpaceTitle(genieSpaceId),
  );

  const isMcpApproval =
    part.callProviderMetadata?.databricks?.approvalRequestId != null;
  const mcpServerName =
    part.callProviderMetadata?.databricks?.mcpServerName?.toString();

  const approved: boolean | undefined =
    'approval' in part ? part.approval?.approved : undefined;

  const effectiveState = effectiveToolState(part, isLoading);

  if (isMcpApproval) {
    // Approval actions live inside the collapsible, so a pending request has to
    // start open or there is no way to allow or deny it.
    return (
      <McpTool defaultOpen={state === 'approval-requested'}>
        <McpToolHeader
          serverName={mcpServerName}
          toolName={displayName}
          state={effectiveState}
          approved={approved}
        />
        <McpToolContent>
          <McpToolInput input={input} />
          {state === 'approval-requested' && (
            <McpApprovalActions
              onApprove={() =>
                submitApproval({ approvalRequestId: toolCallId, approve: true })
              }
              onDeny={() =>
                submitApproval({
                  approvalRequestId: toolCallId,
                  approve: false,
                })
              }
              isSubmitting={isSubmitting && pendingApprovalId === toolCallId}
            />
          )}
          {state === 'output-available' && output != null && (
            <ToolOutput
              output={
                errorText ? (
                  <div className="rounded border p-2 text-red-500">
                    Error: {errorText}
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap font-mono text-sm">
                    {typeof output === 'string'
                      ? output
                      : JSON.stringify(output, null, 2)}
                  </div>
                )
              }
              errorText={undefined}
            />
          )}
        </McpToolContent>
      </McpTool>
    );
  }

  // Collapsed by default: the header still shows the tool and its status,
  // without putting raw request and response payloads in the transcript.
  return (
    <Tool defaultOpen={false}>
      <ToolHeader type={displayName} state={effectiveState} />
      <ToolContent>
        <ToolInput input={input} />
        {state === 'output-available' && (
          <ToolOutput
            output={
              errorText ? (
                <div className="rounded border p-2 text-red-500">
                  Error: {errorText}
                </div>
              ) : (
                <div className="whitespace-pre-wrap font-mono text-sm">
                  {typeof output === 'string'
                    ? output
                    : JSON.stringify(output, null, 2)}
                </div>
              )
            }
            errorText={undefined}
          />
        )}
      </ToolContent>
    </Tool>
  );
};
