import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

type ToolVisibilityContextValue = {
  showToolCalls: boolean;
  setShowToolCalls: (show: boolean) => void;
};

const ToolVisibilityContext = createContext<ToolVisibilityContextValue | null>(
  null,
);

export function ToolVisibilityProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [showToolCalls, setShowToolCalls] = useState(false);
  const value = useMemo(
    () => ({ showToolCalls, setShowToolCalls }),
    [showToolCalls],
  );

  return (
    <ToolVisibilityContext.Provider value={value}>
      {children}
    </ToolVisibilityContext.Provider>
  );
}

export function useToolVisibility(): ToolVisibilityContextValue {
  const context = useContext(ToolVisibilityContext);
  if (!context) {
    throw new Error(
      'useToolVisibility must be used within ToolVisibilityProvider',
    );
  }
  return context;
}
