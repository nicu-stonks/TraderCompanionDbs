import { useState, useEffect } from 'react';
import { Settings, X } from 'lucide-react';
import { fetchPreferences, updatePreferences } from '../api';
import type { CheckInfo } from '../types';

interface Props {
  onClose: () => void;
}

export function SettingsModal({ onClose }: Props) {
  const [prefs, setPrefs] = useState<Record<string, boolean>>({});
  const [allChecks, setAllChecks] = useState<Record<string, CheckInfo>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchPreferences()
      .then((data) => {
        setPrefs(data.preferences);
        setAllChecks(data.all_checks);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleToggle = (key: string) => {
    setPrefs((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updatePreferences(prefs);
      onClose();
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  const handleSelectAll = (enabled: boolean) => {
    const updated: Record<string, boolean> = {};
    for (const key of Object.keys(allChecks)) {
      updated[key] = enabled;
    }
    setPrefs(updated);
  };

  // Group checks by type
  const violations = Object.entries(allChecks).filter(([, v]) => v.type === 'violation');
  const confirmations = Object.entries(allChecks).filter(([, v]) => v.type === 'confirmation');
  const infos = Object.entries(allChecks).filter(([, v]) => v.type === 'info');

  const severityColor = (severity: string) => {
    if (severity === 'red') return 'text-red-500';
    if (severity === 'green') return 'text-green-500';
    return 'text-muted-foreground';
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Violation/Confirmation Settings
          </h3>
          <button onClick={onClose} className="p-1 hover:bg-muted rounded">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-4">
          {loading ? (
            <p className="text-muted-foreground">Loading...</p>
          ) : (
            <>
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => handleSelectAll(true)}
                  className="px-3 py-1 text-xs border border-border rounded hover:bg-muted"
                >
                  Enable All
                </button>
                <button
                  onClick={() => handleSelectAll(false)}
                  className="px-3 py-1 text-xs border border-border rounded hover:bg-muted"
                >
                  Disable All
                </button>
              </div>

              {/* Violations */}
              <div className="mb-4">
                <h4 className="font-medium text-red-500 mb-2 text-sm">Violations ({violations.length})</h4>
                <div className="space-y-1">
                  {violations.map(([key, check]) => (
                    <label key={key} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-muted/50 px-2 py-1 rounded">
                      <input
                        type="checkbox"
                        checked={prefs[key] ?? true}
                        onChange={() => handleToggle(key)}
                        className="rounded"
                      />
                      <span className={severityColor(check.severity)}>{check.name}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Confirmations */}
              <div className="mb-4">
                <h4 className="font-medium text-green-500 mb-2 text-sm">Confirmations ({confirmations.length})</h4>
                <div className="space-y-1">
                  {confirmations.map(([key, check]) => (
                    <label key={key} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-muted/50 px-2 py-1 rounded">
                      <input
                        type="checkbox"
                        checked={prefs[key] ?? true}
                        onChange={() => handleToggle(key)}
                        className="rounded"
                      />
                      <span className="text-green-500">{check.name}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Info */}
              {infos.length > 0 && (
                <div className="mb-4">
                  <h4 className="font-medium text-muted-foreground mb-2 text-sm">Info / Always Shown ({infos.length})</h4>
                  <div className="space-y-1">
                    {infos.map(([key, check]) => (
                      <label key={key} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-muted/50 px-2 py-1 rounded">
                        <input
                          type="checkbox"
                          checked={prefs[key] ?? true}
                          onChange={() => handleToggle(key)}
                          className="rounded"
                        />
                        <span className="text-muted-foreground">{check.name}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex gap-2 p-4 border-t border-border">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:opacity-90 disabled:opacity-50 text-sm font-medium"
          >
            {saving ? 'Saving...' : 'Save Preferences'}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 border border-border rounded-md hover:bg-muted text-sm"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
