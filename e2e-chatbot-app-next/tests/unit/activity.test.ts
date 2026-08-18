import { expect, test } from '@playwright/test';
import type { ChatMessage } from '@chat-template/core';
import { describeActivity, describeTool } from '../../client/src/lib/activity';

const GENIE_QUERY = 'query_space_01f169d6d34a1233bb6d5ab580b58495';
const GENIE_POLL = 'poll_response_01f169d6d34a1233bb6d5ab580b58495';

function assistant(parts: ChatMessage['parts']): ChatMessage {
  return { id: 'm1', role: 'assistant', parts } as ChatMessage;
}

test.describe('describeTool', () => {
  test('names Genie query and poll tools', () => {
    expect(describeTool(GENIE_QUERY)).toBe('Querying Genie');
    expect(describeTool(GENIE_POLL)).toBe('Querying Genie');
    expect(describeTool('genie-01f169d6d34a1233bb6d5ab580b58495')).toBe(
      'Querying Genie',
    );
  });

  test('names web search and python tools', () => {
    expect(describeTool('web_search')).toBe('Searching the web');
    expect(describeTool('web_search_preview')).toBe('Searching the web');
    expect(describeTool('system__ai__python_exec')).toBe('Running Python');
  });

  test('humanizes an unknown tool', () => {
    expect(describeTool('lookup_customer')).toBe('Running lookup customer');
  });
});

test.describe('describeActivity', () => {
  test('is Thinking when there is no assistant message yet', () => {
    expect(describeActivity(undefined)).toBe('Thinking');
    expect(
      describeActivity({ id: 'u', role: 'user', parts: [] } as ChatMessage),
    ).toBe('Thinking');
  });

  test('names a running Genie tool from the tail of the message', () => {
    expect(
      describeActivity(
        assistant([
          {
            type: 'dynamic-tool',
            toolCallId: 't1',
            toolName: GENIE_QUERY,
            input: {},
            state: 'input-available',
          },
        ] as ChatMessage['parts']),
      ),
    ).toBe('Querying Genie');
  });

  test('is Thinking once the latest tool has finished', () => {
    expect(
      describeActivity(
        assistant([
          {
            type: 'dynamic-tool',
            toolCallId: 't1',
            toolName: GENIE_QUERY,
            input: {},
            state: 'output-available',
            output: {},
          },
        ] as ChatMessage['parts']),
      ),
    ).toBe('Thinking');
  });

  test('stays silent while an approval is waiting on the user', () => {
    expect(
      describeActivity(
        assistant([
          {
            type: 'dynamic-tool',
            toolCallId: 't1',
            toolName: 'slack_post',
            input: {},
            state: 'approval-requested',
          },
        ] as ChatMessage['parts']),
      ),
    ).toBeNull();
  });

  test('hides once assistant text is on the page', () => {
    expect(
      describeActivity(assistant([{ type: 'text', text: 'Toronto' }])),
    ).toBeNull();
    expect(describeActivity(assistant([{ type: 'text', text: '   ' }]))).toBe(
      'Thinking',
    );
  });

  test('is Thinking while reasoning is streaming', () => {
    expect(
      describeActivity(
        assistant([{ type: 'reasoning', text: 'Let me look that up.' }]),
      ),
    ).toBe('Thinking');
  });
});
