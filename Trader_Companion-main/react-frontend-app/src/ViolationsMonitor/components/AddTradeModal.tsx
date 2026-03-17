import { useState } from 'react';
import { Plus, Calendar } from 'lucide-react';

interface Props {
  onAdd: (ticker: string, startDate: string, endDate: string | null, useLatest: boolean) => void;
  onClose: () => void;
}

export function AddTradeModal({ onAdd, onClose }: Props) {
  const [ticker, setTicker] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [useLatest, setUseLatest] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker.trim() || !startDate) return;
    onAdd(
      ticker.trim().toUpperCase(),
      startDate,
      useLatest ? null : endDate || null,
      useLatest
    );
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl p-6 w-full max-w-md"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Plus className="h-5 w-5" />
          Add Position to Monitor
        </h3>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Ticker */}
          <div>
            <label className="block text-sm font-medium mb-1">Ticker Symbol</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL"
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              autoFocus
            />
          </div>

          {/* Start Date */}
          <div>
            <label className="text-sm font-medium mb-1 flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              Buy Date (Start)
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          {/* End Date */}
          <div>
            <div className="flex items-center gap-2 mb-1">
              <label className="text-sm font-medium flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                End Date
              </label>
              <label className="flex items-center gap-1 text-xs text-muted-foreground cursor-pointer ml-auto">
                <input
                  type="checkbox"
                  checked={useLatest}
                  onChange={(e) => setUseLatest(e.target.checked)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      setUseLatest((prev) => !prev);
                    }
                  }}
                  className="rounded"
                />
                Use latest (today)
              </label>
            </div>
            {!useLatest && (
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            )}
            {useLatest && (
              <p className="text-xs text-muted-foreground">End date will always be the latest trading day.</p>
            )}
          </div>

          {/* Buttons */}
          <div className="flex gap-2 pt-2">
            <button
              type="submit"
              disabled={!ticker.trim() || !startDate}
              className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:opacity-90 disabled:opacity-50 text-sm font-medium"
            >
              Add Position
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-border rounded-md hover:bg-muted text-sm"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
