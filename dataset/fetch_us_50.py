"""
US Stock Data Fetcher - 50 Diverse Stocks for Analysis
Fetch → Analyze → Auto-recommend 9 best stocks for DRL training
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import time

# 50 DIVERSE US STOCKS - Coverage across volatility spectrum
US_STOCKS = [
    # MEGA-CAP TECH (High volatility) - 7
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    
    # DEFENSIVE (Low volatility) - 7
    'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD',
    
    # FINANCIALS (Medium volatility) - 8
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA',
    
    # INDUSTRIALS (Medium volatility) - 7
    'BA', 'CAT', 'GE', 'UPS', 'HON', 'MMM', 'LMT',
    
    # ENERGY (High volatility) - 5
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    
    # HEALTHCARE (Mixed) - 5
    'UNH', 'ABT', 'TMO', 'ABBV', 'MRK',
    
    # COMMUNICATION (Medium-high) - 5
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ',
    
    # CONSUMER (High) - 5
    'NKE', 'SBUX', 'HD', 'LOW', 'TGT',
    
    # UTILITIES (Very low) - 1
    'NEE'
]


def fetch_stock(symbol, start='2018-01-01', end='2025-01-01'):
    """Fetch single US stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            print(f"  ❌ {symbol} - No data")
            return None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume_detail'
        })
        
        df = df[['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume_detail']]
        
        print(f"  ✓ {symbol:6s} - {len(df):4d} days")
        return df
        
    except Exception as e:
        print(f"  ❌ {symbol} - Error: {e}")
        return None


def save_stock(df, symbol, output_dir='../trainingset'):
    """Save to CSV"""
    if df is None or df.empty:
        return False
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'{symbol}.csv'), index=False)
    return True


def fetch_sp500_index(output_dir='../trainingset'):
    """Fetch S&P 500 for beta calculation"""
    print("\nFetching S&P 500 Index...")
    try:
        ticker = yf.Ticker('^GSPC')
        df = ticker.history(start='2018-01-01', end='2025-01-01')
        
        if df.empty:
            print("  ❌ No S&P 500 data")
            return False
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date', 'Close': 'close_price'})
        df = df[['date', 'close_price']]
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'GSPC.csv'), index=False)
        print(f"  ✓ S&P 500 - {len(df)} days saved")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("US STOCK FETCHER - 50 DIVERSE STOCKS")
    print("="*70)
    print(f"Total: {len(US_STOCKS)} stocks\n")
    
    # Fetch index
    fetch_sp500_index()
    
    # Fetch stocks
    success = 0
    fail = 0
    
    for symbol in US_STOCKS:
        df = fetch_stock(symbol)
        if df is not None and save_stock(df, symbol):
            success += 1
        else:
            fail += 1
        time.sleep(0.3)  # Rate limiting
    
    print("\n" + "="*70)
    print(f"DONE: {success} success, {fail} failed")
    print("="*70)
    print("\nNext: Run classification/market_threshold_analyzer.py")
