from webull_utils import load_and_login

def test_quotes():
    wb = load_and_login()
    if not wb: return

    symbol = 'AAPL'
    print(f"\n--- Testing Single Quote: {symbol} ---")
    quote = wb.get_quote(stock=symbol)
    print(f"Price: {quote.get('price', 'N/A')}")
    print(f"Close: {quote.get('close', 'N/A')}")

    symbols = ['AAPL', 'TSLA', 'NVDA']
    print(f"\n--- Testing Batch Quotes (Serial): {symbols} ---")
    for s in symbols:
        q = wb.get_quote(stock=s)
        p = q.get('close', q.get('price', 'N/A'))
        print(f"{s}: {p}")

if __name__ == "__main__":
    test_quotes()
