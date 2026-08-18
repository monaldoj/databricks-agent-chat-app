import { PreviewMessage } from './message';
import { ActivityIndicator } from './activity-indicator';
import { memo, useEffect } from 'react';
import equal from 'fast-deep-equal';
import type { UseChatHelpers } from '@ai-sdk/react';
import { useMessages } from '@/hooks/use-messages';
import type { ChatMessage, FeedbackMap } from '@chat-template/core';
import { useDataStream } from './data-stream-provider';
import { Conversation, ConversationContent } from './elements/conversation';
import { ArrowDownIcon } from 'lucide-react';

interface MessagesProps {
  status: UseChatHelpers<ChatMessage>['status'];
  messages: ChatMessage[];
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  addToolApprovalResponse: UseChatHelpers<ChatMessage>['addToolApprovalResponse'];
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  regenerate: UseChatHelpers<ChatMessage>['regenerate'];
  isReadonly: boolean;
  selectedModelId: string;
  feedback?: FeedbackMap;
}

function PureMessages({
  status,
  messages,
  setMessages,
  addToolApprovalResponse,
  sendMessage,
  regenerate,
  isReadonly,
  selectedModelId,
  feedback = {},
}: MessagesProps) {
  const {
    containerRef: messagesContainerRef,
    endRef: messagesEndRef,
    isAtBottom,
    scrollToBottom,
  } = useMessages({
    status,
  });

  const lastMessage = messages[messages.length - 1];
  const lastIsAssistant = lastMessage?.role === 'assistant';
  const showActivity = selectedModelId !== 'chat-model-reasoning';

  useDataStream();

  useEffect(() => {
    if (status === 'submitted') {
      requestAnimationFrame(() => {
        const container = messagesContainerRef.current;
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth',
          });
        }
      });
    }
  }, [status, messagesContainerRef]);

  return (
    <div
      ref={messagesContainerRef}
      className="overscroll-behavior-contain -webkit-overflow-scrolling-touch flex-1 touch-pan-y overflow-y-scroll"
      style={{ overflowAnchor: 'none' }}
    >
      <Conversation className="mx-auto flex min-w-0 max-w-4xl flex-col gap-4 md:gap-6">
        <ConversationContent className="flex flex-col gap-4 px-4 py-4 md:gap-6">
          {messages.map((message, index) => (
            <PreviewMessage
              key={message.id}
              message={message}
              allMessages={messages}
              isLoading={
                status === 'streaming' && messages.length - 1 === index
              }
              setMessages={setMessages}
              addToolApprovalResponse={addToolApprovalResponse}
              sendMessage={sendMessage}
              regenerate={regenerate}
              isReadonly={isReadonly}
              activityStatus={
                showActivity && lastIsAssistant && index === messages.length - 1
                  ? status
                  : undefined
              }
              initialFeedback={feedback[message.id]}
            />
          ))}

          {/* Before the assistant message exists, sit the indicator in the
              same spot the first token will occupy — just under the question. */}
          {showActivity && !lastIsAssistant && (
            <ActivityIndicator status={status} lastMessage={lastMessage} />
          )}

          <div
            ref={messagesEndRef}
            className="min-h-[24px] min-w-[24px] shrink-0"
          />
        </ConversationContent>
      </Conversation>

      {!isAtBottom && (
        <button
          className="-translate-x-1/2 absolute bottom-40 left-1/2 z-10 rounded-full border bg-background p-2 shadow-lg transition-colors hover:bg-muted"
          onClick={() => scrollToBottom('smooth')}
          type="button"
          aria-label="Scroll to bottom"
        >
          <ArrowDownIcon className="size-4" />
        </button>
      )}
    </div>
  );
}

export const Messages = memo(PureMessages, (prevProps, nextProps) => {
  // Always re-render during streaming to ensure incremental token display
  if (prevProps.status === 'streaming' || nextProps.status === 'streaming') {
    return false;
  }

  // The activity indicator keys off status, so a turn ending without changing
  // the message list still has to re-render or the indicator never clears.
  if (prevProps.status !== nextProps.status) return false;
  if (prevProps.selectedModelId !== nextProps.selectedModelId) return false;
  if (prevProps.messages.length !== nextProps.messages.length) return false;
  if (!equal(prevProps.messages, nextProps.messages)) return false;
  if (!equal(prevProps.feedback, nextProps.feedback)) return false;

  return true; // Props are equal, skip re-render
});
