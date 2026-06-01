import { createContext, useContext } from 'react';

export type IOSTab = 'home' | 'markets' | 'ideas' | 'strategies';
export interface IOSPush {
  type: 'symbol' | 'idea' | 'strategy';
  id: string;
}

interface IOSNav {
  tab: IOSTab;
  setTab: (t: IOSTab) => void;
  stack: IOSPush[];
  push: (p: IOSPush) => void;
  pop: () => void;
}

export const IOSNavCtx = createContext<IOSNav>({
  tab: 'home',
  setTab: () => {},
  stack: [],
  push: () => {},
  pop: () => {},
});

export const useIOS = () => useContext(IOSNavCtx);
