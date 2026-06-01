import { createContext, useContext } from 'react';

export type PageId =
  | 'overview'
  | 'markets'
  | 'ideas'
  | 'strategies'
  | 'portfolio'
  | 'research'
  | 'model'
  | 'symbol'
  | 'strategy';

export interface Route {
  page: PageId;
  param?: string;
}

export const NavCtx = createContext<{ route: Route; go: (page: PageId, param?: string) => void }>({
  route: { page: 'overview' },
  go: () => {},
});

export const useNav = () => useContext(NavCtx);
