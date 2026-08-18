import { memo, useEffect, useRef, useState } from 'react';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { ChatMessage } from '@chat-template/core';
import { describeActivity } from '@/lib/activity';
import { Shimmer } from './ui/shimmer';

type ChatStatus = UseChatHelpers<ChatMessage>['status'];

/** Seconds of silence before the elapsed counter appears. */
const ELAPSED_VISIBLE_AFTER_SECONDS = 5;

/**
 * Seconds since the current turn started, reset every time the agent goes idle.
 */
function useElapsedSeconds(active: boolean): number {
  const [seconds, setSeconds] = useState(0);
  const startedAt = useRef<number | null>(null);

  useEffect(() => {
    if (!active) {
      startedAt.current = null;
      setSeconds(0);
      return;
    }

    startedAt.current = Date.now();
    setSeconds(0);
    const interval = setInterval(() => {
      if (startedAt.current !== null) {
        setSeconds(Math.floor((Date.now() - startedAt.current) / 1000));
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [active]);

  return seconds;
}

/**
 * A single always-on "the agent is working" row.
 *
 * The agent writes a sentence, hands off to a Genie space for tens of seconds,
 * comes back, calls another tool, and only then writes the answer. Nothing
 * streams during those handoffs, so without this the UI sits still and looks
 * hung.
 */
export const ActivityIndicator = memo(
  ({
    status,
    lastMessage,
    embedded = false,
  }: {
    status: ChatStatus;
    lastMessage?: ChatMessage;
    /** Sit inside the in-progress assistant message, not as a sibling below it. */
    embedded?: boolean;
  }) => {
    const isActive = status === 'submitted' || status === 'streaming';
    const seconds = useElapsedSeconds(isActive);

    if (!isActive) return null;

    const label =
      status === 'submitted' ? 'Thinking' : describeActivity(lastMessage);
    if (label === null) return null;

    const body = (
      <Shimmer className="flex items-center gap-1.5 text-base">
        <span>{label}…</span>
        {seconds >= ELAPSED_VISIBLE_AFTER_SECONDS && (
          <span className="text-sm tabular-nums">{seconds}s</span>
        )}
      </Shimmer>
    );

    if (embedded) {
      return <div data-testid="message-assistant-loading">{body}</div>;
    }

    return (
      <div
        data-testid="message-assistant-loading"
        className="group/message w-full"
        data-role="assistant"
      >
        <div className="flex items-start justify-start gap-3">{body}</div>
      </div>
    );
  },
);

ActivityIndicator.displayName = 'ActivityIndicator';
