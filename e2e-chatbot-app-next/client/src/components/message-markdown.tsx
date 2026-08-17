import { memo, useMemo } from 'react';
import { Response } from './elements/response';
import { AgentChart } from './agent-chart';
import { Shimmer } from './ui/shimmer';
import { parseChartSpec, splitChartBlocks } from '@/lib/chart-spec';

/**
 * A ```chart block we could not plot, kept behind a disclosure so the JSON
 * never sits in the transcript unless the reader asks for it.
 */
function CollapsedChartSource({ source }: { source: string }) {
  return (
    <details className="not-prose overflow-hidden rounded-xl border">
      <summary className="cursor-pointer px-3 py-2 text-muted-foreground text-sm">
        Chart data
      </summary>
      <pre className="overflow-x-auto border-t bg-muted/40 px-3 py-2 font-mono text-xs">
        {source}
      </pre>
    </details>
  );
}

/**
 * Assistant markdown, with ```chart blocks lifted out and drawn as charts.
 *
 * Everything else falls through to the normal markdown renderer untouched.
 * An incomplete block (still streaming in) shows a placeholder rather than
 * leaking half a JSON object into the transcript.
 */
export const MessageMarkdown = memo(({ children }: { children: string }) => {
  const segments = useMemo(() => splitChartBlocks(children), [children]);

  if (segments.length === 1 && segments[0].kind === 'markdown') {
    return <Response>{children}</Response>;
  }

  return (
    <div className="flex flex-col gap-4">
      {segments.map((segment, index) => {
        const key = `segment-${index}`;

        if (segment.kind === 'markdown') {
          return segment.text.trim() ? (
            <Response key={key}>{segment.text}</Response>
          ) : null;
        }

        if (!segment.complete) {
          return (
            <Shimmer key={key} className="text-sm">
              Building chart…
            </Shimmer>
          );
        }

        const spec = parseChartSpec(segment.source);
        if (!spec) {
          return <CollapsedChartSource key={key} source={segment.source} />;
        }

        return <AgentChart key={key} spec={spec} />;
      })}
    </div>
  );
});

MessageMarkdown.displayName = 'MessageMarkdown';
