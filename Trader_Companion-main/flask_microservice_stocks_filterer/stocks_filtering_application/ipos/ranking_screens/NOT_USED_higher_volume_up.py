import csv
from collections import defaultdict
from datetime import datetime, timedelta

def process_csv(input_file, output_file):
    stocks = defaultdict(list)
    
    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = datetime.strptime(row['Date'].split()[0], '%Y-%m-%d')
            close = float(row['Close'])
            open_price = float(row['Open'])
            volume = int(row['Volume'])
            symbol = row['Symbol']
            
            stocks[symbol].append((date, close > open_price, volume))
    
    # Process each stock
    qualifying_stocks = []
    for symbol, data in stocks.items():
        # Sort data by date in descending order
        sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
        
        # Get the most recent date
        if not sorted_data:
            continue
        most_recent_date = sorted_data[0][0]
        
        # Filter for last 2 months
        two_months_ago = most_recent_date - timedelta(days=60)
        filtered_data = [d for d in sorted_data if d[0] >= two_months_ago]
        
        up_volumes = []
        down_volumes = []
        
        for _, is_up, volume in filtered_data:
            if is_up:
                up_volumes.append(volume)
            else:
                down_volumes.append(volume)
        
        if up_volumes and down_volumes:  # Ensure we have both up and down days
            avg_up_volume = sum(up_volumes) / len(up_volumes)
            avg_down_volume = sum(down_volumes) / len(down_volumes)
            
            if avg_down_volume > 0:  # Avoid division by zero
                percentage_difference = ((avg_up_volume - avg_down_volume) / avg_down_volume) * 100
                
                if percentage_difference >= 50:
                    qualifying_stocks.append((symbol, percentage_difference))
    
    # Write qualifying stocks to output CSV
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Up_Volume_Difference'])
        for symbol, percentage in qualifying_stocks:
            writer.writerow([symbol, f"{percentage:.2f}%"])

    print(f"Volume up Analysis complete. {len(qualifying_stocks)} qualifying stocks saved to {output_file}")

# Usage
input_file = './ranking_screens/passed_stocks_input_data/filtered_price_data.csv'
output_file = './ranking_screens/results/higher_volume_up.csv'
process_csv(input_file, output_file)