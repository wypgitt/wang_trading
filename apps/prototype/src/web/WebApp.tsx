import { useNav } from '../nav';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { OverviewPage } from './pages/OverviewPage';
import { MarketsPage } from './pages/MarketsPage';
import { IdeasPage } from './pages/IdeasPage';
import { StrategiesPage } from './pages/StrategiesPage';
import { StrategyDetailPage } from './pages/StrategyDetailPage';
import { SymbolDetailPage } from './pages/SymbolDetailPage';
import { PortfolioPage } from './pages/PortfolioPage';
import { ResearchPage } from './pages/ResearchPage';
import { ModelPage } from './pages/ModelPage';

export function WebApp() {
  const { route } = useNav();

  let page;
  switch (route.page) {
    case 'overview':
      page = <OverviewPage />;
      break;
    case 'markets':
      page = <MarketsPage />;
      break;
    case 'ideas':
      page = <IdeasPage />;
      break;
    case 'strategies':
      page = <StrategiesPage />;
      break;
    case 'strategy':
      page = <StrategyDetailPage id={route.param ?? ''} />;
      break;
    case 'symbol':
      page = <SymbolDetailPage symbol={route.param ?? ''} />;
      break;
    case 'portfolio':
      page = <PortfolioPage />;
      break;
    case 'research':
      page = <ResearchPage />;
      break;
    case 'model':
      page = <ModelPage />;
      break;
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'var(--sidebar-w) 1fr', height: '100%', background: 'var(--bg-0)' }}>
      <Sidebar />
      <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0 }}>
        <TopBar />
        <div className="scroll-y" style={{ flex: 1, minHeight: 0 }}>
          <div style={{ maxWidth: 1480, margin: '0 auto', padding: '24px 30px 64px' }}>
            <div key={route.page + (route.param ?? '')} className="fade-up">
              {page}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
