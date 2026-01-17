
import yfinance as yf
import pandas as pd
import re
import numpy as np

def resolve_ticker(ticker):
    if not ticker:
        return "2330.TW"
    ticker = ticker.strip().upper()
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return ticker
    if "." in ticker:
        base = ticker.split(".")[0]
        if re.match(r'^\d{4,6}$', base):
            return f"{base}.TW"
        return ticker
    if re.match(r'^\d{4,6}$', ticker):
        return f"{ticker}.TW"
    return ticker

def get_signal_score_mock(data, rsi_val):
    # Simplified mock of the actual logic
    score_pts = 0
    current_price = float(data['Close'].iloc[-1])
    ema20 = float(data['Close'].ewm(span=20).mean().iloc[-1])
    if current_price > ema20: score_pts += 1
    if 50 < rsi_val < 75: score_pts += 1
    return score_pts

def test_full_analysis(ticker):
    resolved = resolve_ticker(ticker)
    print(f"\n--- Testing {ticker} (Resolved: {resolved}) ---")
    
    try:
        # 1. Price Data
        d = yf.download(resolved, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if (d is None or d.empty) and (".TW" in resolved or ".TWO" in resolved):
            alt = resolved.replace(".TW", ".TWO") if ".TW" in resolved else resolved.replace(".TWO", ".TW")
            d = yf.download(alt, period="6mo", interval="1d", progress=False, auto_adjust=True)
            if not d.empty: resolved = alt
        
        if d.empty:
            print(f"  [FAILED] Price data not found.")
            return False
        
        print(f"  [OK] Price data retrieved ({len(d)} rows).")

        # 2. Financials
        stock = yf.Ticker(resolved)
        financials = stock.financials
        if financials is None or financials.empty:
            financials = stock.quarterly_financials
        
        if financials is not None and not financials.empty:
            print(f"  [OK] Financial data retrieved ({len(financials.columns)} periods).")
        else:
            print(f"  [WARN] No financial data (Expected for some indices/ETFs).")

        # 3. Technicals (RSI)
        delta = d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        print(f"  [OK] RSI calculated: {current_rsi:.2f}")

        # 4. Signal Score
        score = get_signal_score_mock(d, current_rsi)
        print(f"  [OK] Signal score mock: {score}")

        return True
    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return False

test_tickers = [
    "2330", "8069", "AAPL", "NVDA", "BTC-USD", "^TWII", "GC=F", "TWD=X", "EURUSD=X"
]

print("Starting deep stock functionality tests...")
results = []
for t in test_tickers:
    success = test_full_analysis(t)
    results.append((t, success))

print("\nDeep Test Summary:")
for t, s in results:
    status = "FULL SUPPORT" if s else "PARTIAL/FAILED"
    print(f"{t}: {status}")

