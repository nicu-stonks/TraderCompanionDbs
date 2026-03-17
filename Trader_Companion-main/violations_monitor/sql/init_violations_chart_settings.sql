CREATE TABLE IF NOT EXISTS chart_settings (
    ticker TEXT PRIMARY KEY,
    sma_settings_json TEXT NOT NULL,
    weekly_sma_settings_json TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'close',
    highlight_marker_gap INTEGER NOT NULL DEFAULT 0,
    open_on_bars INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chart_settings_ticker
ON chart_settings(ticker);
