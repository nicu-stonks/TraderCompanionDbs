import { useState, ReactNode } from 'react';
import { Info, X, BookOpen } from 'lucide-react';
import { GLOSSARY, TREND_UP_DEFINITION } from '../types';

interface Props {
  onClose: () => void;
}

/** Convert a raw glossary key like "good_bad_close" to display title "Good Bad Close" */
const formatKey = (key: string): string =>
  key.replace(/_/g, ' ').replace(/pct/gi, '%').replace(/\b\w/g, (c) => c.toUpperCase());

/** Highlight all occurrences of `query` inside `text` with a yellow background */
function Highlight({ text, query }: { text: string; query: string }): ReactNode {
  if (!query) return text;
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);
  if (parts.length === 1) return text;
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <mark key={i} className="bg-yellow-400/40 text-inherit rounded-sm px-px">{part}</mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

export function GlossaryModal({ onClose }: Props) {
  const [search, setSearch] = useState('');

  const entries = Object.entries(GLOSSARY).filter(([key, desc]) => {
    const q = search.toLowerCase();
    if (!q) return true;
    const title = formatKey(key).toLowerCase();
    return title.includes(q) || desc.toLowerCase().includes(q);
  });

  const severityForKey = (key: string): string => {
    // Color-code based on common patterns in the key name
    const violations = [
      'closes_on_low', 'down_', 'largest_', 'bearish', 'squat', 'rs_new_low',
      'widest', 'close_below', 'very_extended', 'weekly_lower', 'daily_lower',
      'daily_megaphone', 'down_3pct', 'big_down_day', 'large_squat', 'violations_count',
    ];
    const confirmations = [
      'closes_on_high', 'bullish', 'rs_new_high', 'up_30pct', 'inside_day',
      'close_above', 'weekly_higher', 'daily_higher', 'up_3pct', 'big_up_day', 'ants',
      'confirmations_count',
    ];
    if (confirmations.some((p) => key.startsWith(p) || key.includes(p)))
      return 'border-l-green-500 bg-green-500/5';
    if (violations.some((p) => key.startsWith(p) || key.includes(p)))
      return 'border-l-red-500 bg-red-500/5';
    return 'border-l-gray-500 bg-muted/30';
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-full max-w-2xl max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Violations & Confirmations Glossary
          </h3>
          <button onClick={onClose} className="p-1 hover:bg-muted rounded">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-4 border-b border-border">
          <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-md text-sm mb-3">
            <p className="font-medium text-blue-600 dark:text-blue-400 mb-1">
              <Info className="h-4 w-4 inline mr-1" /> What is a "Trend Up"?
            </p>
            {TREND_UP_DEFINITION.map((paragraph, i) => (
              <p key={i} className="text-muted-foreground text-xs mb-2 last:mb-0">
                {paragraph}
              </p>
            ))}
          </div>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search checks..."
            className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>

        <div className="overflow-y-auto flex-1 p-4 space-y-2">
          {entries.map(([key, desc]) => (
            <div
              key={key}
              className={`border-l-4 rounded-r-md p-3 ${severityForKey(key)}`}
            >
              <h4 className="font-medium text-sm">
                <Highlight text={formatKey(key)} query={search} />
              </h4>
              <p className="text-xs text-muted-foreground mt-0.5"><Highlight text={desc} query={search} /></p>
            </div>
          ))}
          {entries.length === 0 && (
            <p className="text-muted-foreground text-sm text-center py-4">No matching checks found.</p>
          )}
        </div>

        <div className="p-3 border-t border-border flex justify-between text-xs text-muted-foreground">
          <span className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-red-500/50" /> Violation
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-green-500/50" /> Confirmation
            </span>
          </span>
          <button onClick={onClose} className="px-3 py-1 border border-border rounded hover:bg-muted">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
