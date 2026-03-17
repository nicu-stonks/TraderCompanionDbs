// components/Navigation.tsx
import { Link, useLocation } from 'react-router-dom';
import { ThemeToggle } from './ThemeToggle';

export function Navigation() {
  const location = useLocation();
  const currentPage = location.pathname.slice(1) || 'stocks_screeners';

  return (
    <nav className="w-full bg-background border-b">
      <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 relative">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 py-2 pr-14">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Link
              to="/personal_ranking"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'personal_ranking'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Personal Ranking
            </Link>
            <Link
              to="/trade_history"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'trade_history'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Trade History
            </Link>
            <Link
              to="/trading_stats"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'trading_stats'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Trading Stats
            </Link>
            <Link
              to="/post_analysis"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'post_analysis'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Post Analysis
            </Link>
            <Link
              to="/ticker_management"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'ticker_management'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Ticker Monitoring
            </Link>
            <Link
              to="/custom_orders"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'custom_orders'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Custom Orders
            </Link>
            <Link
              to="/price_alerts"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'price_alerts'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Price Alerts
            </Link>
            <Link
              to="/stocks_screeners"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'stocks_screeners'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Stocks Screeners
            </Link>
            <Link
              to="/nutrients-tracker"
              className={`inline-flex items-center px-4 py-2 text-sm font-medium transition-colors hover:text-primary
                ${currentPage === 'nutrients-tracker'
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted-foreground'
                }`}
            >
              Nutrients Tracker
            </Link>

          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}