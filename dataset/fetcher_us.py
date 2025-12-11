"""
US Stock Data Fetcher - 50 Diverse Stocks for Threshold Analysis
Auto-select 9 best stocks after analysis (3 Low + 3 Medium + 3 High Risk)
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import time

# 50 DIVERSE US STOCKS - Coverage across volatility spectrum
US_STOCKS = [
    # === MEGA-CAP TECH (High volatility potential) - 7 stocks ===
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    
    # === LARGE-CAP DEFENSIVE (Low volatility expected) - 7 stocks ===
    'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD',
    
    # === FINANCIALS (Medium volatility) - 8 stocks ===
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA',
    
    # === INDUSTRIALS (Medium volatility) - 7 stocks ===
    'BA', 'CAT', 'GE', 'UPS', 'HON', 'MMM', 'LMT',
    
    # === ENERGY (High volatility) - 5 stocks ===
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    
    # === HEALTHCARE (Mixed volatility) - 5 stocks ===
    'UNH', 'ABT', 'TMO', 'ABBV', 'MRK',
    
    # === COMMUNICATION (Medium-high volatility) - 5 stocks ===
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ',
    
    # === CONSUMER DISCRETIONARY (High volatility) - 5 stocks ===
    'NKE', 'SBUX', 'HD', 'LOW', 'TGT',
    
    # === UTILITIES (Very low volatility) - 1 stock ===
    'NEE'
]


class USStockFetcher:
    def __init__(self):
        self.output_dir = "../trainingset"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_us_stocks(self):
            # Tech/Growth
            'SQ', 'SNAP', 'TWLO', 'ZM', 'DOCU', 'ROKU', 'PINS', 'U',
            # Finance
            'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE', 'NDAQ',
            # Industrial
            'EMR', 'ITW', 'ETN', 'ROK', 'PH', 'AME', 'DOV', 'FTV',
            # Healthcare
            'BIIB', 'REGN', 'VRTX', 'ILMN', 'ALGN', 'IDXX', 'IQV', 'A',
            # Consumer
            'DPZ', 'YUM', 'CMG', 'DNKN', 'QSR', 'WING', 'TXRH', 'DRI'
        ]
        
        # === BANKING & FINANCE (Tương đương Banking VN) - 30 stocks ===
        banking = [
            # Regional Banks
            'KEY', 'RF', 'CFG', 'HBAN', 'FITB', 'MTB', 'ZION', 'CMA',
            # Investment/Brokerage
            'IBKR', 'MKTX', 'VIRT', 'LPLA', 'SF', 'BGC', 'LAZ', 'EVR',
            # Insurance
            'PGR', 'ALL', 'TRV', 'CB', 'AIG', 'MET', 'PRU', 'AFL',
            # Asset Management
            'TROW', 'BEN', 'IVZ', 'AMG', 'APAM', 'SEIC'
        ]
        
        # === INDUSTRIAL & MANUFACTURING (Tương đương Industrial VN) - 30 stocks ===
        industrial = [
            # Manufacturing
            'CMI', 'PCAR', 'IR', 'XYL', 'GNRC', 'ALLE', 'AOS', 'CARR',
            # Materials
            'DD', 'DOW', 'LYB', 'APD', 'ECL', 'SHW', 'PPG', 'NUE',
            # Construction
            'MLM', 'VMC', 'BLD', 'BLDR', 'MTZ', 'OC', 'SUM', 'UFPI',
            # Transportation
            'FDX', 'DAL', 'UAL', 'LUV', 'AAL', 'JBLU'
        ]
        
        # === TECH & GROWTH (Tương đương Tech VN) - 30 stocks ===
        tech_growth = [
            # Software/Cloud
            'NOW', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'S', 'ESTC',
            # Cybersecurity
            'PANW', 'FTNT', 'CHKP', 'CYBR', 'OKTA', 'RPD', 'TENB', 'QLYS',
            # Semiconductors
            'MU', 'AMAT', 'LRCX', 'KLAC', 'MCHP', 'MRVL', 'SWKS', 'QRVO',
            # E-commerce/Digital
            'SHOP', 'EBAY', 'ETSY', 'W', 'MELI', 'SE'
        ]
        
        # === INDEX FOR COMPARISON ===
        indices = ['^GSPC', '^DJI', '^IXIC']  # S&P500, Dow, Nasdaq
        
        # Combine all
        all_stocks = blue_chips + large_cap + mid_cap + banking + industrial + tech_growth + indices
        
        # Remove duplicates
        unique_stocks = list(set(all_stocks))
        
        print(f"Total US stocks selected: {len(unique_stocks)}")
        print(f"  - Blue Chips: {len(blue_chips)}")
        print(f"  - Large Cap: {len(large_cap)}")
        print(f"  - Mid Cap: {len(mid_cap)}")
        print(f"  - Banking: {len(banking)}")
        print(f"  - Industrial: {len(industrial)}")
        print(f"  - Tech/Growth: {len(tech_growth)}")
        
        return unique_stocks
    
    def fetch_data(self, symbol, start_date='2018-01-01', end_date='2025-01-01'):
        """Fetch historical data for one symbol"""
        try:
            print(f"  Downloading {symbol}...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # Rename columns to match VN format
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Close': 'close_price',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Volume': 'volume_detail'
            })
            
            # Select only needed columns
            df = df[['date', 'close_price', 'open_price', 'high_price', 'low_price', 'volume_detail']]
            
            # Sort by date descending (like VN data)
            df = df.sort_values('date', ascending=False)
            
            return df
            
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_all_stocks(self):
        """Fetch all selected US stocks"""
        stocks = self.get_us_stocks()
        print(f"\nStarting to fetch {len(stocks)} US stocks...")
        print("="*70)
        
        successful = []
        failed = []
        
        for i, symbol in enumerate(stocks, 1):
            print(f"[{i}/{len(stocks)}] {symbol}")
            
            df = self.fetch_data(symbol)
            
            if not df.empty and len(df) > 100:
                # Save with .csv extension
                filename = f"{self.output_dir}/{symbol.replace('^', 'INDEX_')}.csv"
                df.to_csv(filename, index=False)
                successful.append(symbol)
                print(f"  ✓ Saved {len(df)} records")
            else:
                failed.append(symbol)
                print(f"  ✗ Failed or insufficient data")
            
            # Rate limiting
            time.sleep(0.2)
        
        print("\n" + "="*70)
        print(f"COMPLETED:")
        print(f"  ✓ Successful: {len(successful)}")
        print(f"  ✗ Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailed symbols: {', '.join(failed)}")
        
        return successful, failed


if __name__ == "__main__":
    print("="*70)
    print("US STOCK DATA FETCHER FOR DRL TRADING MODEL")
    print("="*70)
    print("\nThis script will download ~200 US stocks equivalent to VN market")
    print("Data period: 2018-01-01 to 2025-01-01")
    print("\nRequirement: pip install yfinance")
    print("="*70)
    
    input("\nPress Enter to start downloading...")
    
    fetcher = USStockFetcher()
    successful, failed = fetcher.fetch_all_stocks()
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print(f"Data saved to: {fetcher.output_dir}/")
    print("="*70)
