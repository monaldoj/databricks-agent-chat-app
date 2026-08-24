import type { ChatMessage } from '@chat-template/core';
import type { TextUIPart } from 'ai';

/**
 * Creates segments of parts that can be rendered as a single component.
 *
 * `source-url` parts are omitted here. Gemini google_search emits them as soon
 * as grounding arrives, often before the rest of the answer, so the message
 * renderer collects them separately and shows them at the bottom of the turn.
 */
export const createMessagePartSegments = (parts: ChatMessage['parts']) => {
  // An array of arrays of parts
  // Allows us to render multiple parts as a single component
  const out: ChatMessage['parts'][] = [];
  for (const part of parts) {
    if (part.type === 'source-url') continue;

    const lastBlock = out[out.length - 1] || null;

    if (
      lastBlock?.[0]?.type === 'text' &&
      part.type === 'text' &&
      !isNamePart(part) &&
      !isNamePart(lastBlock[0])
    ) {
      // If the text part, or the previous part contains a <name></name> tag, add it to a new block
      // Otherwise, append sequential text parts to the same block
      lastBlock.push(part);
    }
    // Otherwise, add the current part to a new block
    else {
      out.push([part]);
    }
  }

  return out;
};

export const isNamePart = (
  part: ChatMessage['parts'][number],
): part is TextUIPart => {
  return (
    part.type === 'text' &&
    part.text?.startsWith('<name>') &&
    part.text?.endsWith('</name>')
  );
};
export const formatNamePart = (part: ChatMessage['parts'][number]) => {
  if (!isNamePart(part)) return null;
  return part.text?.replace('<name>', '').replace('</name>', '');
};

/**
 * Takes a segment of parts and joins them into a markdown-formatted string.
 * Citations are rendered in a footer, not inlined into this markdown.
 */
export const joinMessagePartSegments = (parts: ChatMessage['parts']) => {
  return parts.reduce((acc, part) => {
    if (part.type === 'text') {
      return acc + part.text;
    }
    return acc;
  }, '');
};
