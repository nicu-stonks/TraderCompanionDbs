import csv
import sys
from datetime import datetime
import os.path


def read_banned_symbols(file_path):
    if not os.path.exists(file_path):
        return {}

    banned_dict = {}
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            banned_dict[row['Symbol']] = {
                'date': datetime.strptime(row['Date'], '%Y-%m-%d'),
                'duration': int(row['BanDurationInWeeks'])
            }
    return banned_dict


def write_banned_symbols(file_path, banned_symbols):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Symbol', 'BanDurationInWeeks'])
        for symbol, ban_info in banned_symbols.items():
            writer.writerow([
                ban_info['date'].strftime('%Y-%m-%d'),
                symbol,
                str(ban_info['duration'])
            ])


def add_banned_symbols(file_path, new_bans):
    # Read existing bans
    banned_symbols = read_banned_symbols(file_path)
    current_date = datetime.now()

    # Add or update each new ban
    for symbol, duration in new_bans:
        banned_symbols[symbol] = {
            'date': current_date,
            'duration': int(duration)
        }

    # Write back to file
    write_banned_symbols(file_path, banned_symbols)
    return len(new_bans)


def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python add_banned_stocks.py TICKER1 DURATION1 [TICKER2 DURATION2 ...]")
        print("Example: python add_banned_stocks.py AAPL 3 MSFT 1")
        sys.exit(1)

    # Parse command line arguments into pairs
    new_bans = []
    for i in range(1, len(sys.argv), 2):
        ticker = sys.argv[i].upper()
        try:
            duration = int(sys.argv[i + 1])
            if duration <= 0:
                raise ValueError("Duration must be positive")
            new_bans.append((ticker, duration))
        except ValueError as e:
            print(f"Error: Invalid duration for {ticker}. {str(e)}")
            sys.exit(1)

    import os

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Define the banned stocks file path
    file_path = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "banned_stocks", "banned_stocks.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Add the bans
    added_count = add_banned_symbols(file_path, new_bans)

    # Print summary
    print(f"Added/Updated {added_count} stock(s) to ban list:")
    for symbol, duration in new_bans:
        print(f"- {symbol}: {duration} month(s)")


if __name__ == "__main__":
    main()