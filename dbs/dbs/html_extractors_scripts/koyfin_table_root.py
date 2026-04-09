import csv
import io
from bs4 import BeautifulSoup

def parse_html(input_text: str) -> str:
    if not input_text or not input_text.strip():
        return ""
        
    try:
        soup = BeautifulSoup(input_text, 'html.parser')
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 1. Extract Headers and map valid columns
        headers = []
        valid_indices = []
        
        header_row = soup.find('div', class_=lambda c: c and 'headerRow' in c)
        if header_row:
            # Target the specific cell wrappers to prevent duplicate text extraction
            header_cells = header_row.find_all('div', class_=lambda c: c and 'headerCell' in c)
            
            for i, cell in enumerate(header_cells):
                text = cell.get_text(strip=True)
                
                # Index 0 is an empty sticky cell used for the sparkline charts
                if i == 0 and not text:
                    text = "Chart"
                    
                # Ignore the Current/LTM column
                if "Current/LTM" not in text:
                    headers.append(text)
                    valid_indices.append(i)
            
            # Filter out the empty chart column from the headers and our tracking indices
            final_headers = [h for h in headers if h != "Chart"]
            valid_data_indices = [i for i in valid_indices if i != 0]
            
            if final_headers:
                writer.writerow(final_headers)

        # 2. Extract Data Rows
        rows = soup.find_all('div', class_=lambda c: c and 'faTable__row' in c)
        for row in rows:
            row_data = []
            
            # Extract all grid cells sequentially 
            all_cells = row.find_all('div', class_=lambda c: c and 'cellGrid' in c)
            
            for cell in all_cells:
                val = cell.get_text(separator="", strip=True)
                row_data.append(val)
            
            # Apply our valid indices filter to drop the chart and LTM columns
            filtered_row = [row_data[i] for i in valid_data_indices if i < len(row_data)]
            
            # Only write the row if it has actual data beyond the metric name
            if any(filtered_row) and any(filtered_row[1:]):
                writer.writerow(filtered_row)

        return output.getvalue().strip()
        
    except Exception as e:
        return f"Error extracting data: {str(e)}"