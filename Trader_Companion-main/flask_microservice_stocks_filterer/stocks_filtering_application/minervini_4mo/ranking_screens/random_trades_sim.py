import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from datetime import datetime, timedelta
import sys
import os

def simulate_trade(stock_data, start_date, symbol):
    """
    Simulate a single trade starting from start_date
    Returns: (profit_loss_percentage, days_held, exit_reason)
    """
    # Handle timezone compatibility
    start_date_ts = pd.Timestamp(start_date)
    
    # If stock data has timezone but start_date doesn't, localize start_date
    if stock_data['Date'].dt.tz is not None and start_date_ts.tz is None:
        start_date_ts = start_date_ts.tz_localize(stock_data['Date'].dt.tz)
    # If stock data has timezone and start_date has different timezone, convert
    elif stock_data['Date'].dt.tz is not None and start_date_ts.tz is not None:
        start_date_ts = start_date_ts.tz_convert(stock_data['Date'].dt.tz)
    
    # Filter data from start_date onwards
    trade_data = stock_data[stock_data['Date'] >= start_date_ts].sort_values('Date')
    
    if len(trade_data) == 0:
        return None, None, "No data available"
    
    # Entry price (first available price after start_date)
    entry_price = trade_data.iloc[0]['Close']
    entry_date = trade_data.iloc[0]['Date']
    
    # Simulate trade day by day
    for i, row in trade_data.iterrows():
        current_price = row['Close']
        current_date = row['Date']
        
        # Calculate percentage change from entry
        pct_change = (current_price - entry_price) / entry_price * 100
        
        # Check exit conditions
        if pct_change <= -6.0:  # Stop loss at -6%
            days_held = (current_date - entry_date).days
            return pct_change, days_held, "Stop Loss"
        elif pct_change >= 12.0:  # Take profit at +12%
            days_held = (current_date - entry_date).days
            return pct_change, days_held, "Take Profit"
    
    # If we reach here, we held until end of data
    final_price = trade_data.iloc[-1]['Close']
    final_pct_change = (final_price - entry_price) / entry_price * 100
    days_held = (trade_data.iloc[-1]['Date'] - entry_date).days
    return final_pct_change, days_held, "End of Data"

def run_trading_simulation(input_file, num_trades):
    # Read the CSV file
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Get unique symbols
    symbols = df['Symbol'].unique()
    print(f"Found {len(symbols)} unique symbols")
    
    # Define date range (5 months ago to now)
    end_date = datetime.now()
    start_date_range = end_date - timedelta(days=150)
    
    # Convert to pandas timestamps for easier timezone handling
    end_date = pd.Timestamp(end_date)
    start_date_range = pd.Timestamp(start_date_range)
    
    # Initialize tracking variables
    initial_capital = 10000  # Starting with $10,000
    current_capital = initial_capital
    trades_log = []
    
    print(f"\nStarting simulation with {num_trades} trades...")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("-" * 80)
    
    successful_trades = 0
    
    for trade_num in range(1, num_trades + 1):
        # Step 1: Pick random date between now and 5 months ago
        random_days = random.randint(0, 150)
        random_start_date = start_date_range + timedelta(days=random_days)
        
        # Step 2: Pick random stock
        random_symbol = random.choice(symbols)
        
        # Get stock data for this symbol
        stock_data = df[df['Symbol'] == random_symbol].copy()
        
        # Step 3: Simulate trade
        pct_return, days_held, exit_reason = simulate_trade(stock_data, random_start_date, random_symbol)
        
        if pct_return is not None:
            # Calculate dollar return
            trade_amount = current_capital
            dollar_return = trade_amount * (pct_return / 100)
            current_capital += dollar_return
            
            # Log the trade
            trade_info = {
                'trade_num': trade_num,
                'symbol': random_symbol,
                'start_date': random_start_date.strftime('%Y-%B-%d'),
                'pct_return': pct_return,
                'dollar_return': dollar_return,
                'days_held': days_held,
                'exit_reason': exit_reason,
                'capital_after': current_capital
            }
            trades_log.append(trade_info)
            successful_trades += 1
            
            # Print trade details
            print(f"Trade #{trade_num:3d} | {random_symbol:6s} | {random_start_date.strftime('%Y-%B-%d')} | "
                  f"{pct_return:+6.2f}% | ${dollar_return:+8.2f} | {days_held:3d} days | "
                  f"{exit_reason:12s} | Capital: ${current_capital:10.2f}")
        else:
            print(f"Trade #{trade_num:3d} | {random_symbol:6s} | {random_start_date.strftime('%Y-%B-%d')} | "
                  f"FAILED - No data available")
    
    print("-" * 80)
    print(f"Simulation Complete!")
    print(f"Successful Trades: {successful_trades}/{num_trades}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${current_capital:,.2f}")
    print(f"Total Return: {((current_capital - initial_capital) / initial_capital * 100):+.2f}%")
    
    if successful_trades > 0:
        # Calculate statistics
        returns = [trade['pct_return'] for trade in trades_log]
        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
        avg_return = np.mean(returns)
        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Return per Trade: {avg_return:+.2f}%")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        
        # Create plots
        create_plots(trades_log)
    
    return trades_log

def create_plots(trades_log):
    """Create visualization plots of the trading results"""
    if not trades_log:
        print("No trades to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trading Simulation Results', fontsize=16)
    
    # Plot 1: Capital over time
    trade_nums = [trade['trade_num'] for trade in trades_log]
    capital_values = [trade['capital_after'] for trade in trades_log]
    
    ax1.plot(trade_nums, capital_values, 'b-', linewidth=2)
    ax1.set_title('Capital Over Time')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Distribution of returns
    returns = [trade['pct_return'] for trade in trades_log]
    ax2.hist(returns, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Distribution of Trade Returns')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Cumulative returns
    cumulative_returns = []
    cumulative = 0
    for trade in trades_log:
        cumulative += trade['pct_return']
        cumulative_returns.append(cumulative)
    
    ax3.plot(trade_nums, cumulative_returns, 'g-', linewidth=2)
    ax3.set_title('Cumulative Returns')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Days held distribution
    days_held = [trade['days_held'] for trade in trades_log if trade['days_held'] is not None]
    if days_held:
        ax4.hist(days_held, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Distribution of Days Held')
        ax4.set_xlabel('Days Held')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Trading Simulation Script')
    parser.add_argument('num_trades', type=int, help='Number of trades to simulate')
    
    args = parser.parse_args()
    
    if args.num_trades <= 0:
        print("Error: Number of trades must be positive")
        sys.exit(1)
    
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)
    
    # Append the correct relative path to the input file
    input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Run the simulation
    trades_log = run_trading_simulation(input_file, args.num_trades)

if __name__ == '__main__':
    main()