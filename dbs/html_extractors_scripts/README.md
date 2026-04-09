# HTML Extractor Scripts

This folder stores user-provided Python extractor scripts used by the Biggest Winners "Fill With AI" modal.

Expected script pattern (recommended):

```python
def parse_html(input_text: str) -> str:
    # Parse HTML and return extracted text/CSV/JSON as a string
    return "..."
```

Supported callable names:
- `parse_html`
- `extract_html`
- `extract`
- `parse`
- `main`

The selected script is executed server-side and receives pasted HTML input text.
The returned output is shown in UI and can be appended to the generated AI prompt.
