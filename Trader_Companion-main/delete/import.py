import csv

# Input and output file names
input_file = 'coq.csv'
output_file = 'output_symbols.csv'

# Open the input CSV file
with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    
    # Open the output CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['Symbol'])
        
        # Write each symbol without quotes
        for row in reader:
            symbol = row['Symbol'].replace('"', '')
            writer.writerow([symbol])

print(f"Cleaned symbols saved to {output_file}")
