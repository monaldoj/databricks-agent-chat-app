import type { ChatMessage } from '@chat-template/core';
import type {
  AnchorHTMLAttributes,
  ComponentType,
  PropsWithChildren,
  ReactNode,
} from 'react';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { cn } from '@/lib/utils';

/**
 * ReactMarkdown/Streamdown component that handles Databricks message citations.
 *
 * @example
 * <Streamdown components={{ a: DatabricksMessageCitationStreamdownIntegration }} />
 */
export const DatabricksMessageCitationStreamdownIntegration: ComponentType<
  AnchorHTMLAttributes<HTMLAnchorElement>
> = (props) => {
  if (isDatabricksMessageCitationLink(props.href)) {
    return (
      <DatabricksMessageCitationRenderer
        {...props}
        href={decodeDatabricksMessageCitationLink(props.href)}
      />
    );
  }
  return <DefaultAnchor {...props} />;
};

export type SourcePart = Extract<
  ChatMessage['parts'][number],
  { type: 'source-url' }
>;

/**
 * Unique web citations from a message, in the order they first appeared.
 *
 * Gemini google_search (and other grounded answers) often emit `source-url`
 * parts right after the search step. The transcript still has more to say, so
 * callers render this list at the bottom of the assistant turn instead of
 * inlining the pills where they arrived.
 */
export function collectSourceUrlParts(
  parts: ChatMessage['parts'],
): SourcePart[] {
  const seen = new Set<string>();
  const collected: SourcePart[] = [];
  for (const part of parts) {
    if (part.type !== 'source-url' || !part.url || seen.has(part.url)) {
      continue;
    }
    seen.add(part.url);
    collected.push(part);
  }
  return collected;
}

// Adds a unique suffix to the link to indicate that it is a Databricks message citation.
const encodeDatabricksMessageCitationLink = (part: SourcePart) =>
  `${part.url}::databricks_citation`;

// Removes the unique suffix from the link to get the original link.
const decodeDatabricksMessageCitationLink = (link: string) =>
  link.replace('::databricks_citation', '');

// Creates a markdown link to the Databricks message citation.
export const createDatabricksMessageCitationMarkdown = (part: SourcePart) =>
  `[${part.title || part.url}](${encodeDatabricksMessageCitationLink(part)})`;

// Checks if the link is a Databricks message citation.
const isDatabricksMessageCitationLink = (
  link?: string,
): link is `${string}::databricks_citation` =>
  link?.endsWith('::databricks_citation') ?? false;

const PILL_CLASS =
  'my-0.5 mr-1 inline-flex max-w-[16rem] items-center truncate rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 align-middle text-xs font-medium text-blue-600 no-underline hover:bg-blue-100 hover:text-blue-800 dark:border-blue-800 dark:bg-blue-950/60 dark:text-blue-400 dark:hover:bg-blue-900/70 dark:hover:text-blue-300';

/**
 * Compact label for a source pill: a page title when we have one, otherwise
 * the hostname so a raw URL does not wrap across the transcript.
 */
export function sourceLinkLabel(
  label: string | undefined,
  href: string,
): string {
  const text = label?.trim();
  try {
    const host = new URL(href).hostname.replace(/^www\./, '');
    if (!text || text === href || /^https?:\/\//i.test(text)) {
      return host || text || href;
    }
    return text;
  } catch {
    return text || href;
  }
}

function textFromChildren(children: ReactNode): string {
  if (typeof children === 'string' || typeof children === 'number') {
    return String(children);
  }
  if (Array.isArray(children)) {
    return children.map(textFromChildren).join('');
  }
  return '';
}

export function SourceLinkPill({
  href,
  children,
  className,
}: PropsWithChildren<{
  href: string;
  className?: string;
}>) {
  const label = sourceLinkLabel(textFromChildren(children) || undefined, href);

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(PILL_CLASS, className)}
    >
      <span className="truncate">{label}</span>
    </a>
  );
}

/** Citation pills collected at the end of an assistant turn. */
export function MessageSources({ sources }: { sources: SourcePart[] }) {
  if (sources.length === 0) return null;

  return (
    <div data-testid="message-sources" className="flex flex-col gap-1.5">
      <div className="font-medium text-muted-foreground text-xs">Sources</div>
      <div className="flex flex-wrap items-center">
        {sources.map((source) => (
          <SourceLinkPill key={source.url} href={source.url}>
            {source.title || source.url}
          </SourceLinkPill>
        ))}
      </div>
    </div>
  );
}

// Renders the Databricks message citation.
const DatabricksMessageCitationRenderer = (
  props: PropsWithChildren<{
    href: string;
  }>,
) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <SourceLinkPill href={props.href}>{props.children}</SourceLinkPill>
      </TooltipTrigger>
      <TooltipContent
        style={{ maxWidth: '300px', padding: '8px', wordWrap: 'break-word' }}
      >
        {props.href}
      </TooltipContent>
    </Tooltip>
  );
};

const DefaultAnchor: ComponentType<AnchorHTMLAttributes<HTMLAnchorElement>> = ({
  className,
  children,
  href,
  ...props
}) => {
  const isIncomplete = href === 'streamdown:incomplete-link';
  const isFootnoteLink = href?.startsWith('#');
  const isExternalHttp = typeof href === 'string' && /^https?:\/\//i.test(href);

  if (!isIncomplete && !isFootnoteLink && isExternalHttp) {
    return <SourceLinkPill href={href}>{children}</SourceLinkPill>;
  }

  return (
    <a
      className={cn(
        'font-medium text-blue-600 underline hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300',
        className,
      )}
      data-incomplete={isIncomplete}
      data-streamdown="link"
      href={href}
      {...props}
      {...(isFootnoteLink
        ? {
            target: '_self',
          }
        : {
            target: '_blank',
            rel: 'noopener noreferrer',
          })}
    >
      {children}
    </a>
  );
};
