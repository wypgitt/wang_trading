import { useState } from 'react';
import { Icon } from '../components/Icon';
import { IOSNavCtx, IOSPush, IOSTab } from './iosNav';
import { Home } from './screens/Home';
import { Markets } from './screens/Markets';
import { Ideas } from './screens/Ideas';
import { Strategies } from './screens/Strategies';
import { SymbolScreen } from './screens/SymbolScreen';
import { IdeaScreen } from './screens/IdeaScreen';
import { StrategyScreen } from './screens/StrategyScreen';
import { C } from '../lib/colors';

const TABS: { id: IOSTab; label: string; icon: string }[] = [
  { id: 'home', label: 'Home', icon: 'home' },
  { id: 'markets', label: 'Markets', icon: 'markets' },
  { id: 'ideas', label: 'Ideas', icon: 'ideas' },
  { id: 'strategies', label: 'Strategies', icon: 'strategies' },
];

function TabBar({ tab, onTab }: { tab: IOSTab; onTab: (t: IOSTab) => void }) {
  return (
    <div
      style={{
        flex: 'none',
        height: 82,
        paddingBottom: 22,
        display: 'flex',
        borderTop: '1px solid var(--border)',
        background: 'rgba(12,15,20,0.86)',
        backdropFilter: 'blur(18px)',
      }}
    >
      {TABS.map((t) => {
        const on = t.id === tab;
        return (
          <button key={t.id} data-tab={t.id} onClick={() => onTab(t.id)} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4, color: on ? C.accent2 : C.text3 }}>
            <Icon name={t.icon} size={23} />
            <span style={{ fontSize: 10.5, fontWeight: 600 }}>{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export function IOSApp() {
  const [tab, setTabState] = useState<IOSTab>('home');
  const [stack, setStack] = useState<IOSPush[]>([]);
  const top = stack[stack.length - 1];

  const setTab = (t: IOSTab) => {
    setStack([]);
    setTabState(t);
  };
  const push = (p: IOSPush) => setStack((s) => [...s, p]);
  const pop = () => setStack((s) => s.slice(0, -1));

  let content;
  if (top) {
    if (top.type === 'symbol') content = <SymbolScreen id={top.id} />;
    else if (top.type === 'idea') content = <IdeaScreen id={top.id} />;
    else content = <StrategyScreen id={top.id} />;
  } else {
    const tabScreen = tab === 'home' ? <Home /> : tab === 'markets' ? <Markets /> : tab === 'ideas' ? <Ideas /> : <Strategies />;
    content = (
      <div className="scroll-y" style={{ flex: 1, minHeight: 0 }}>
        {tabScreen}
      </div>
    );
  }

  const routeKey = top ? `${top.type}:${top.id}` : tab;

  return (
    <IOSNavCtx.Provider value={{ tab, setTab, stack, push, pop }}>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <div key={routeKey} className="fade-up" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {content}
        </div>
        <TabBar tab={tab} onTab={setTab} />
      </div>
    </IOSNavCtx.Provider>
  );
}
