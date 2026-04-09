import csv
import io
from bs4 import BeautifulSoup

def parse_html(input_text: str) -> str:
    soup = BeautifulSoup(input_text, 'html.parser')
    
    # 1. Extract Headers accurately without duplication
    headers = []
    header_cells = soup.find_all('div', class_=lambda c: c and 'headerCell' in c)
    for cell in header_cells:
        txt = cell.get_text(strip=True)
        # Only add if text exists (skips empty top-left corner cells)
        if txt:
            headers.append(txt)
            
    # Map valid columns and explicitly drop 'Current/LTM'
    keep_indices = []
    clean_headers = []
    for i, h in enumerate(headers):
        if 'current/ltm' not in h.lower():
            keep_indices.append(i)
            clean_headers.append(h)

    # 2. Extract Data Rows
    rows = []
    # Find all actual data rows, ignoring 'add-ticker' interactive elements
    row_nodes = soup.find_all('div', class_=lambda c: c and 'base-table-row__root' in c and 'add-ticker' not in c)
    
    for row in row_nodes:
        cells = row.find_all('div', class_=lambda c: c and 'cellGrid' in c)
        row_vals = []
        
        for cell in cells:
            txt = cell.get_text(strip=True)
            # Skip cells that are just structural SVGs without text
            if not txt and cell.find('svg'):
                continue
            row_vals.append(txt)
            
        if len(row_vals) > 1:
            clean_row = []
            # Align row values precisely with our kept header indices
            for i in keep_indices:
                if i < len(row_vals):
                    clean_row.append(row_vals[i])
                else:
                    clean_row.append("") # padding for missing cells safely
                    
            rows.append(clean_row)

    # 3. Format output as CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    if clean_headers:
        writer.writerow(clean_headers)
    writer.writerows(rows)
    
    return output.getvalue().strip()