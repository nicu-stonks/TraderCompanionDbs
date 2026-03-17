import React, { memo } from 'react';
import { Alert as AlertType } from '../types';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Trash2, Power } from 'lucide-react';

interface AlertsTableProps {
  alerts: AlertType[];
  priceDirections: Map<number, 'up' | 'down'>;
  onToggleActive: (id: number, isActive: boolean) => Promise<void>;
  onDelete: (id: number) => void | Promise<void>;
  onDeleteAll: () => void;
  monitoredTickers?: string[]; // Optional - for comparing with alert tickers
}

const AlertsTableComponent: React.FC<AlertsTableProps> = ({ alerts, onToggleActive, onDelete, onDeleteAll }) => {

  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A';
    return `$${price.toFixed(2)}`;
  };

  const getPriceDifference = (current: number | null, alert: number) => {
    if (current === null) return null;
    // Express difference as "how far the alert is from the current price".
    // If alert is above current => positive (green). If alert is below current => negative (red).
    const diff = alert - current;
    const pct = (alert === 0 ? 0 : (diff / alert) * 100).toFixed(2);
    return { diff, pct };
  };

  // Curated palette — distinct hues, all readable on light & dark backgrounds
  const tickerColors = [
    '#e06c75', // red
    '#d19a66', // orange
    '#e5c07b', // gold
    '#98c379', // green
    '#56b6c2', // teal
    '#61afef', // blue
    '#c678dd', // purple
    '#f0a1c2', // pink
    '#be5046', // rust
    '#c08aff', // lavender
  ];

  // Deterministic color per ticker — uses a better-spreading hash
  const getTickerColor = (ticker: string): string => {
    let hash = 5381;
    for (let i = 0; i < ticker.length; i++) {
      hash = ((hash << 5) + hash + ticker.charCodeAt(i)) | 0; // djb2
    }
    return tickerColors[Math.abs(hash) % tickerColors.length];
  };

  if (alerts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Active Alerts</CardTitle>
          <CardDescription>No alerts created yet</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <CardTitle>Price Alerts</CardTitle>
            <Badge variant="secondary" className="text-xs font-medium">
              {alerts.length} alert{alerts.length !== 1 ? 's' : ''}
            </Badge>
          </div>
          <CardDescription>
            Monitor your ticker price alerts
          </CardDescription>
        </div>
        <Button
          variant="destructive"
          size="sm"
          onClick={onDeleteAll}
          title="Delete all alerts"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete All
        </Button>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table className="whitespace-nowrap">
            <TableHeader>
              <TableRow>
                <TableHead>Ticker</TableHead>
                <TableHead>Current(% Chg)</TableHead>
                <TableHead>Alert Price(% Diff)</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...alerts].sort((a, b) => {
                // 1. Sort by ticker name alphabetically
                const tickerCmp = a.ticker.localeCompare(b.ticker);
                if (tickerCmp !== 0) return tickerCmp;
                // 2. Within same ticker, sort by alert price % diff descending
                const diffA = a.current_price !== null ? ((a.alert_price - a.current_price) / a.alert_price) * 100 : 0;
                const diffB = b.current_price !== null ? ((b.alert_price - b.current_price) / b.alert_price) * 100 : 0;
                return diffB - diffA; // descending
              }).map((alert) => {
                const priceDiff = getPriceDifference(alert.current_price, alert.alert_price);

                return (
                  <TableRow key={alert.id}>
                    <TableCell className="font-medium" style={{ color: getTickerColor(alert.ticker) }}>{alert.ticker}</TableCell>
                    <TableCell>
                      <span className="font-semibold text-foreground">
                        {formatPrice(alert.current_price)}
                      </span>
                      {alert.percent_change !== null && (
                        <span
                          className={`ml-2 font-semibold tabular-nums ${alert.percent_change >= 0
                            ? 'text-green-600 dark:text-green-500'
                            : 'text-red-500 dark:text-red-400'
                            }`}
                        >
                          ({alert.percent_change >= 0 ? '+' : ''}{alert.percent_change.toFixed(2)}%)
                        </span>
                      )}
                      {!alert.is_active && alert.current_price !== null && (
                        <span className="ml-2 text-xs text-muted-foreground" title="Last known price (alert inactive)">
                          (last)
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      <span className="font-semibold text-foreground">
                        {formatPrice(alert.alert_price)}
                      </span>
                      {priceDiff && (
                        <span
                          className={`ml-2 font-semibold tabular-nums ${priceDiff.diff >= 0
                            ? 'text-green-600 dark:text-green-500'
                            : 'text-red-500 dark:text-red-400'
                            }`}
                        >
                          ({priceDiff.diff >= 0 ? '+' : ''}{priceDiff.pct}%)
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {alert.triggered ? (
                        <Badge variant="destructive">Triggered</Badge>
                      ) : alert.is_active ? (
                        <Badge variant="default" className="bg-green-700 dark:bg-green-800">Active</Badge>
                      ) : (
                        <Badge variant="secondary">Inactive</Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            if (alert.is_active || alert.triggered) {
                              onToggleActive(alert.id, false);
                            }
                          }}
                          title={alert.is_active || alert.triggered ? 'Stop alarm' : 'Alert already stopped - can only be deleted'}
                          disabled={!alert.is_active && !alert.triggered}
                        >
                          {(alert.is_active || alert.triggered) ? (
                            <Power className="h-4 w-4" />
                          ) : (
                            <Power className="h-4 w-4 opacity-30" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onDelete(alert.id)}
                          title="Delete"
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};

// Memoize to prevent re-renders when sibling components update
export const AlertsTable = memo(AlertsTableComponent);
