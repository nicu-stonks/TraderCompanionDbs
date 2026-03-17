"""
IBD Relative Strength Chart
Displays stock OHLC price bars, S&P 500 line, and IBD-style Relative Strength line
Interactive controls: Arrow keys, scroll wheel, and buttons to adjust visible days
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from datetime import datetime, timedelta


class InteractiveRSChart:
    def __init__(self, ticker_symbol: str, period: str = "2y"):
        self.ticker_symbol = ticker_symbol
        self.period = period
        self.min_days = 30
        self.days_step = 20
        
        # Fetch all data
        self.stock_data, self.sp500_data = self.fetch_data()
        self.total_days = len(self.stock_data)
        self.visible_days = self.total_days  # Start with all data visible
        
        # Calculate RS metrics for full dataset
        self.rs_line_full = self.calculate_rs_line(self.stock_data['Close'], self.sp500_data['Close'])
        self.rs_score_full = self.calculate_ibd_rs_score(self.stock_data['Close'], self.sp500_data['Close'])
        
        # Setup the chart
        self.setup_chart()
        self.update_chart()
        
    def calculate_performance(self, prices: pd.Series, trading_days: int) -> pd.Series:
        """Calculate percentage performance over specified trading days."""
        return (prices / prices.shift(trading_days) - 1) * 100

    def calculate_ibd_rs_score(self, stock_prices: pd.Series, sp500_prices: pd.Series) -> pd.Series:
        """
        Calculate IBD-style Relative Strength Score.
        Formula: RS = (40% * 3-month perf) + (20% * 6-month perf) + (20% * 9-month perf) + (20% * 12-month perf)
        """
        DAYS_3M, DAYS_6M, DAYS_9M, DAYS_12M = 63, 126, 189, 252
        
        stock_perf_3m = self.calculate_performance(stock_prices, DAYS_3M)
        stock_perf_6m = self.calculate_performance(stock_prices, DAYS_6M)
        stock_perf_9m = self.calculate_performance(stock_prices, DAYS_9M)
        stock_perf_12m = self.calculate_performance(stock_prices, DAYS_12M)
        
        sp500_perf_3m = self.calculate_performance(sp500_prices, DAYS_3M)
        sp500_perf_6m = self.calculate_performance(sp500_prices, DAYS_6M)
        sp500_perf_9m = self.calculate_performance(sp500_prices, DAYS_9M)
        sp500_perf_12m = self.calculate_performance(sp500_prices, DAYS_12M)
        
        rel_perf_3m = stock_perf_3m - sp500_perf_3m
        rel_perf_6m = stock_perf_6m - sp500_perf_6m
        rel_perf_9m = stock_perf_9m - sp500_perf_9m
        rel_perf_12m = stock_perf_12m - sp500_perf_12m
        
        rs_score = (0.40 * rel_perf_3m) + (0.20 * rel_perf_6m) + (0.20 * rel_perf_9m) + (0.20 * rel_perf_12m)
        return rs_score

    def calculate_rs_line(self, stock_prices: pd.Series, sp500_prices: pd.Series) -> pd.Series:
        """Calculate the simple RS Line (stock price / S&P 500 price ratio)."""
        rs_ratio = stock_prices / sp500_prices
        rs_line = (rs_ratio / rs_ratio.iloc[0]) * 100
        return rs_line

    def fetch_data(self) -> tuple:
        """Fetch stock and S&P 500 data from yfinance."""
        print(f"Fetching data for {self.ticker_symbol} and S&P 500...")
        
        stock = yf.Ticker(self.ticker_symbol)
        stock_data = stock.history(period=self.period)
        
        sp500 = yf.Ticker("SPY")
        sp500_data = sp500.history(period=self.period)
        
        common_dates = stock_data.index.intersection(sp500_data.index)
        stock_data = stock_data.loc[common_dates]
        sp500_data = sp500_data.loc[common_dates]
        
        print(f"Loaded {len(stock_data)} trading days of data")
        print(f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        
        return stock_data, sp500_data

    def normalize_series(self, series: pd.Series, base: float = 100) -> pd.Series:
        """Normalize a series to start at a given base value."""
        if len(series) == 0:
            return series
        return (series / series.iloc[0]) * base

    def draw_ohlc_bars(self, ax, dates, opens, highs, lows, closes, first_close):
        """Draw OHLC price bars (not candlesticks)."""
        # Normalize prices
        opens_norm = (opens / first_close) * 100
        highs_norm = (highs / first_close) * 100
        lows_norm = (lows / first_close) * 100
        closes_norm = (closes / first_close) * 100
        
        bar_width = 0.4  # Width of horizontal ticks
        
        for i, date in enumerate(dates):
            date_num = mdates.date2num(date)
            
            # Determine color based on close vs open
            if closes.iloc[i] >= opens.iloc[i]:
                color = '#26a69a'  # Green for bullish
            else:
                color = '#ef5350'  # Red for bearish
            
            # Draw vertical line (high to low)
            ax.plot([date_num, date_num], [lows_norm.iloc[i], highs_norm.iloc[i]], 
                   color=color, linewidth=1.2)
            
            # Draw left horizontal tick (open)
            ax.plot([date_num - bar_width, date_num], 
                   [opens_norm.iloc[i], opens_norm.iloc[i]], 
                   color=color, linewidth=1.2)
            
            # Draw right horizontal tick (close)
            ax.plot([date_num, date_num + bar_width], 
                   [closes_norm.iloc[i], closes_norm.iloc[i]], 
                   color=color, linewidth=1.2)

    def setup_chart(self):
        """Setup the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create gridspec for layout
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.3], hspace=0.25)
        
        self.ax1 = self.fig.add_subplot(gs[0])  # Price chart
        self.ax2 = self.fig.add_subplot(gs[1])  # RS Score
        button_ax = self.fig.add_subplot(gs[2])  # Button area
        button_ax.axis('off')
        
        # Create secondary y-axis for RS Line
        self.ax1_rs = self.ax1.twinx()
        
        # Setup buttons
        self.btn_less = Button(plt.axes([0.3, 0.02, 0.12, 0.04]), '◀ Less Days (-20)')
        self.btn_more = Button(plt.axes([0.58, 0.02, 0.12, 0.04]), 'More Days (+20) ▶')
        self.btn_reset = Button(plt.axes([0.44, 0.02, 0.12, 0.04]), 'Reset All')
        
        # Connect button events
        self.btn_less.on_clicked(self.on_less_days)
        self.btn_more.on_clicked(self.on_more_days)
        self.btn_reset.on_clicked(self.on_reset)
        
        # Connect keyboard and scroll events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.fig.suptitle(f'{self.ticker_symbol} vs S&P 500 - IBD Relative Strength Analysis', 
                         fontsize=16, fontweight='bold')

    def update_chart(self):
        """Update the chart with current visible days."""
        # Clear axes
        self.ax1.clear()
        self.ax1_rs.clear()
        self.ax2.clear()
        
        # Get visible data slice (most recent N days)
        start_idx = max(0, self.total_days - self.visible_days)
        stock_visible = self.stock_data.iloc[start_idx:]
        sp500_visible = self.sp500_data.iloc[start_idx:]
        rs_line_visible = self.rs_line_full.iloc[start_idx:]
        rs_score_visible = self.rs_score_full.iloc[start_idx:]
        
        if len(stock_visible) == 0:
            return
        
        # Normalize for visible period
        first_close = stock_visible['Close'].iloc[0]
        stock_normalized = self.normalize_series(stock_visible['Close'])
        sp500_normalized = self.normalize_series(sp500_visible['Close'])
        rs_line_normalized = self.normalize_series(rs_line_visible)
        
        # Draw OHLC price bars
        self.draw_ohlc_bars(
            self.ax1,
            stock_visible.index,
            stock_visible['Open'],
            stock_visible['High'],
            stock_visible['Low'],
            stock_visible['Close'],
            first_close
        )
        
        # Plot S&P 500 line
        self.ax1.plot(stock_visible.index, sp500_normalized, color='#2196f3', 
                     linewidth=1.5, label='S&P 500', alpha=0.7)
        
        # Plot RS Line on secondary axis
        self.ax1_rs.plot(stock_visible.index, rs_line_normalized, color='#9c27b0', 
                        linewidth=2, label='RS Line', linestyle='-')
        self.ax1_rs.axhline(y=100, color='#9c27b0', linestyle='--', alpha=0.3, linewidth=1)
        self.ax1_rs.set_ylabel('RS Line', color='#9c27b0', fontsize=12)
        self.ax1_rs.tick_params(axis='y', labelcolor='#9c27b0')
        
        # Formatting for main price axis
        self.ax1.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, self.visible_days // 120)))
        self.ax1.grid(True, alpha=0.3)
        
        # Create custom legend for price bars
        legend_elements = [
            Line2D([0], [0], color='#26a69a', linewidth=2, label=f'{self.ticker_symbol} (Bullish)'),
            Line2D([0], [0], color='#ef5350', linewidth=2, label=f'{self.ticker_symbol} (Bearish)'),
            Line2D([0], [0], color='#2196f3', linewidth=2, label='S&P 500'),
        ]
        self.ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
        self.ax1_rs.legend(loc='upper right', fontsize=10)
        
        # Calculate stats for title
        latest_rs = rs_line_normalized.iloc[-1] if len(rs_line_normalized) > 0 else 0
        latest_stock_pct = ((stock_visible['Close'].iloc[-1] / stock_visible['Close'].iloc[0]) - 1) * 100
        latest_sp500_pct = ((sp500_visible['Close'].iloc[-1] / sp500_visible['Close'].iloc[0]) - 1) * 100
        
        self.ax1.set_title(
            f"{self.ticker_symbol}: {latest_stock_pct:+.1f}%  |  S&P 500: {latest_sp500_pct:+.1f}%  |  "
            f"RS Line: {latest_rs:.1f}  |  Showing {len(stock_visible)} of {self.total_days} days",
            fontsize=11, pad=10
        )
        
        # Plot RS Score
        self.ax2.fill_between(stock_visible.index, rs_score_visible, 0,
                             where=(rs_score_visible >= 0), color='#4caf50', alpha=0.5, 
                             label='Outperforming S&P')
        self.ax2.fill_between(stock_visible.index, rs_score_visible, 0,
                             where=(rs_score_visible < 0), color='#f44336', alpha=0.5, 
                             label='Underperforming S&P')
        self.ax2.plot(stock_visible.index, rs_score_visible, color='#333333', linewidth=1)
        self.ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        self.ax2.set_ylabel('IBD RS Score', fontsize=12)
        self.ax2.set_xlabel('Date', fontsize=12)
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, self.visible_days // 120)))
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper left', fontsize=10)
        
        valid_rs = rs_score_visible.dropna()
        if len(valid_rs) > 0:
            latest_rs_score = valid_rs.iloc[-1]
            self.ax2.set_title(f'IBD Weighted RS Score: {latest_rs_score:+.2f}', fontsize=11, pad=10)
        
        # Rotate x-axis labels
        for ax in [self.ax1, self.ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        self.fig.canvas.draw_idle()

    def on_less_days(self, event):
        """Remove days from view."""
        if self.visible_days > self.min_days:
            self.visible_days = max(self.min_days, self.visible_days - self.days_step)
            self.update_chart()

    def on_more_days(self, event):
        """Add days to view."""
        if self.visible_days < self.total_days:
            self.visible_days = min(self.total_days, self.visible_days + self.days_step)
            self.update_chart()

    def on_reset(self, event):
        """Reset to show all days."""
        self.visible_days = self.total_days
        self.update_chart()

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'left':
            self.on_less_days(event)
        elif event.key == 'right':
            self.on_more_days(event)
        elif event.key == 'r':
            self.on_reset(event)

    def on_scroll(self, event):
        """Handle mouse scroll events."""
        if event.button == 'up':
            self.on_less_days(event)
        elif event.button == 'down':
            self.on_more_days(event)

    def show(self):
        """Display the chart."""
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.12)
        plt.show()

    def save(self, filename=None):
        """Save the chart to a file."""
        if filename is None:
            filename = f'{self.ticker_symbol}_rs_chart.png'
        self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved as '{filename}'")


if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (default: AAPL): ").strip().upper() or "AAPL"
    
    print("\n" + "="*60)
    print(f"Generating IBD Relative Strength Chart for {ticker}")
    print("="*60 + "\n")
    
    chart = InteractiveRSChart(ticker, period="2y")
    chart.save()
    
    print("\n" + "="*60)
    print("CONTROLS:")
    print("="*60)
    print("• Left Arrow / Scroll Up: Remove days (zoom in to recent)")
    print("• Right Arrow / Scroll Down: Add days (zoom out)")
    print("• 'R' key: Reset to show all days")
    print("• Buttons: Click to add/remove days")
    print("="*60)
    print("\nINTERPRETATION GUIDE:")
    print("="*60)
    print("• OHLC Bars: Left tick=Open, Right tick=Close, Vertical=High-Low")
    print("• Green bars: Close >= Open (bullish)")
    print("• Red bars: Close < Open (bearish)")
    print("• RS Line (purple): When rising, stock outperforms S&P 500")
    print("• RS Score > 0 (green): Stock outperforming S&P")
    print("• RS Score < 0 (red): Stock underperforming S&P")
    print("="*60)
    
    chart.show()
