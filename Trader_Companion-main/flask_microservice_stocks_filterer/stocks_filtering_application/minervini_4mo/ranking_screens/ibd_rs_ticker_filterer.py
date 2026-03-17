import pandas as pd
import os
import numpy as np

def main():
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Define file paths for input and output
    rs_raw_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results", "raw_rs_file.csv")
    obligatory_passed_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results", "obligatory_passed_stocks.csv")
    not_banned_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "banned_stocks", "stocks_not_banned.csv")
    
    # Output files
    filtered_rs_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results", "filtered_rs_file.csv")
    filtered_banned_rs_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "ranking_screens", "results", "filtered_banned_rs_file.csv")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filtered_rs_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print file paths for debugging
    print(f"Raw RS file: {rs_raw_file}")
    print(f"Obligatory passed stocks file: {obligatory_passed_file}")
    print(f"Not banned stocks file: {not_banned_file}")
    print(f"Output filtered RS file: {filtered_rs_file}")
    print(f"Output filtered banned RS file: {filtered_banned_rs_file}")
    
    # Read the input files
    print("Reading input files...")
    
    # Read RS ratings file
    try:
        rs_df = pd.read_csv(rs_raw_file)
        print(f"Loaded {len(rs_df)} stocks with RS ratings")
    except FileNotFoundError:
        print(f"Error: Raw RS file not found at {rs_raw_file}")
        return
    
    # Read obligatory passed stocks file
    try:
        obligatory_df = pd.read_csv(obligatory_passed_file)
        obligatory_stocks = set(obligatory_df['Symbol'].tolist())
        print(f"Loaded {len(obligatory_stocks)} obligatory passed stocks")
    except FileNotFoundError:
        print(f"Error: Obligatory passed stocks file not found at {obligatory_passed_file}")
        return
    
    # Read not banned stocks file
    try:
        not_banned_df = pd.read_csv(not_banned_file)
        not_banned_stocks = set(not_banned_df['Symbol'].tolist())
        print(f"Loaded {len(not_banned_stocks)} not banned stocks")
    except FileNotFoundError:
        print(f"Error: Not banned stocks file not found at {not_banned_file}")
        return
    
    # Step 1: Filter RS file to only include obligatory passed stocks
    filtered_rs_df = rs_df[rs_df['Symbol'].isin(obligatory_stocks)]
    print(f"After filtering for obligatory passed stocks: {len(filtered_rs_df)} stocks remain")
    
    # Save filtered RS file
    filtered_rs_df.to_csv(filtered_rs_file, index=False)
    print(f"Saved filtered RS ratings to {filtered_rs_file}")
    
    # Step 2: Further filter to exclude banned stocks
    filtered_banned_rs_df = filtered_rs_df[filtered_rs_df['Symbol'].isin(not_banned_stocks)]
    print(f"After filtering out banned stocks: {len(filtered_banned_rs_df)} stocks remain")
    
    # Save filtered banned RS file
    filtered_banned_rs_df.to_csv(filtered_banned_rs_file, index=False)
    print(f"Saved filtered non-banned RS ratings to {filtered_banned_rs_file}")

if __name__ == "__main__":
    main()