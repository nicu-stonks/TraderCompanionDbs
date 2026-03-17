import csv
from datetime import datetime, timedelta
import os


def create_empty_not_banned_file(file_path):
    """Create a new empty stocks_not_banned.csv file with just the header."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Symbol'])


def read_symbols(file_path):
    with open(file_path, 'r') as f:
        return [row['Symbol'] for row in csv.DictReader(f)]


def read_banned_symbols(file_path):
    banned_dict = {}
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            banned_dict[row['Symbol']] = {
                'date': datetime.strptime(row['Date'], '%Y-%m-%d'),
                'duration': int(row['BanDurationInWeeks']) * 7  # Convert months to days
            }
    return banned_dict


def is_still_banned(ban_info, current_date):
    return current_date < ban_info['date'] + timedelta(days=ban_info['duration'])


def write_banned_symbols(file_path, banned_symbols):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Symbol', 'BanDurationInWeeks'])
        for symbol, ban_info in banned_symbols.items():
            writer.writerow([
                ban_info['date'].strftime('%Y-%m-%d'),
                symbol,
                str(ban_info['duration'] // 7)  # Convert days back to months
            ])


def process_symbols(symbols, banned_symbols, current_date):
    initial_ban_count = len(banned_symbols)

    # Check all banned symbols and remove expired bans
    expired_bans = [
        symbol for symbol, ban_info in banned_symbols.items()
        if not is_still_banned(ban_info, current_date)
    ]

    for symbol in expired_bans:
        del banned_symbols[symbol]

    # Get allowed symbols (not in banned list)
    allowed_symbols = [symbol for symbol in symbols if symbol not in banned_symbols]

    return allowed_symbols, banned_symbols, len(expired_bans)


def main():
    current_date = datetime.now()
    
    import os

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Define the file paths dynamically
    not_banned_file_path = os.path.join(script_dir, "stocks_filtering_application", "ipos", "banned_stocks", "stocks_not_banned.csv")
    passed_stocks_file_path = os.path.join(script_dir, "stocks_filtering_application", "ipos","obligatory_screens", "results", "obligatory_passed_stocks.csv")
    banned_stocks_file_path = os.path.join(script_dir, "stocks_filtering_application", "ipos","banned_stocks", "banned_stocks.csv")

    # Ensure the necessary directories exist
    os.makedirs(os.path.dirname(not_banned_file_path), exist_ok=True)
    
    # Create a new empty stocks_not_banned.csv file
    print("Creating new empty stocks_not_banned.csv file...")
    create_empty_not_banned_file(not_banned_file_path)

    try:
        # Read input files
        symbols = read_symbols(passed_stocks_file_path)
        banned_symbols = read_banned_symbols(banned_stocks_file_path)

        # Process symbols
        allowed_symbols, updated_banned_symbols, removed_count = process_symbols(symbols, banned_symbols, current_date)

        # Update banned_stocks.csv to remove expired bans
        write_banned_symbols(banned_stocks_file_path, updated_banned_symbols)

        # Write the allowed symbols to stocks_not_banned.csv
        with open(not_banned_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Symbol'])
            for symbol in allowed_symbols:
                writer.writerow([symbol])

        print(f"Checking {len(symbols)} symbols for ban. {len(allowed_symbols)} symbols are allowed.")
        print(f"Removed {removed_count} expired bans.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Ensure the file exists even if there's an error
        create_empty_not_banned_file(not_banned_file_path)


if __name__ == "__main__":
    main()