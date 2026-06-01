import { useState } from 'react';
import { NavCtx, Route } from './nav';
import { PrototypeBar } from './components/PrototypeBar';
import { WebApp } from './web/WebApp';
import { IOSStage } from './ios/IOSStage';

export function App() {
  const [platform, setPlatform] = useState<'web' | 'ios'>('web');
  const [route, setRoute] = useState<Route>({ page: 'overview' });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <PrototypeBar platform={platform} setPlatform={setPlatform} />
      <div style={{ flex: 1, minHeight: 0 }}>
        {platform === 'web' ? (
          <NavCtx.Provider value={{ route, go: (page, param) => setRoute({ page, param }) }}>
            <WebApp />
          </NavCtx.Provider>
        ) : (
          <IOSStage />
        )}
      </div>
    </div>
  );
}
