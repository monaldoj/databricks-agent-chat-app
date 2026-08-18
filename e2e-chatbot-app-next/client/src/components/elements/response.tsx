import { type ComponentProps, memo } from 'react';
import { DatabricksMessageCitationStreamdownIntegration } from '../databricks-message-citation';
import { Streamdown } from 'streamdown';

type ResponseProps = ComponentProps<typeof Streamdown>;

export const Response = memo(
  (props: ResponseProps) => {
    return (
      <Streamdown
        components={{
          a: DatabricksMessageCitationStreamdownIntegration,
        }}
        className="flex flex-col gap-4"
        {...props}
        // Streaming mode updates parsed blocks via useTransition, which
        // defers paints until the turn goes idle — the answer then pops
        // in all at once. Static mode renders the growing string immediately.
        mode="static"
      />
    );
  },
  (prevProps, nextProps) => prevProps.children === nextProps.children,
);

Response.displayName = 'Response';
