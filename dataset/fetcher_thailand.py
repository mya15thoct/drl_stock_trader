import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class ThailandStockFetcher:
    """Fetcher for Thailand stock market data using Yahoo Finance"""
    
    def __init__(self, output_dir='thailand_dataset'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def fetch_stock(self, symbol, start_date=None, end_date=None):
        """
        Fetch stock data from Yahoo Finance for Thailand stocks
        
        Args:
            symbol: Stock symbol (e.g., 'PTT.BK' for PTT on Bangkok Stock Exchange)
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: today)
        
        Returns:
            DataFrame with stock data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Add .BK suffix if not present (Bangkok Stock Exchange)
            if not symbol.endswith('.BK'):
                symbol_yf = f"{symbol}.BK"
            else:
                symbol_yf = symbol
            
            print(f"Fetching {symbol_yf} from {start_date} to {end_date}...")
            
            stock = yf.Ticker(symbol_yf)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data found for {symbol_yf}")
                return None
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Save to CSV
            base_symbol = symbol.replace('.BK', '')
            output_path = os.path.join(self.output_dir, f"{base_symbol}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} records to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols, start_date=None, end_date=None):
        """
        Fetch multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary of {symbol: DataFrame}
        """
        results = {}
        for symbol in symbols:
            df = self.fetch_stock(symbol, start_date, end_date)
            if df is not None:
                base_symbol = symbol.replace('.BK', '')
                results[base_symbol] = df
        
        return results
    
    def fetch_set_index(self, start_date=None, end_date=None):
        """
        Fetch SET Index (Stock Exchange of Thailand Index)
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with SET Index data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            print(f"Fetching SET Index from {start_date} to {end_date}...")
            
            # SET Index symbol on Yahoo Finance
            index = yf.Ticker("^SET.BK")
            df = index.history(start=start_date, end=end_date)
            
            if df.empty:
                print("Warning: No data found for SET Index")
                return None
            
            df = df.reset_index()
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, "SET.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} records to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching SET Index: {str(e)}")
            return None


if __name__ == "__main__":
    # Example usage
    fetcher = ThailandStockFetcher()
    
    # Fetch some major Thailand stocks
    major_stocks = [
        'PTT',      # PTT Public Company Limited
        'KBANK',    # Kasikornbank
        'SCB',      # Siam Commercial Bank
        'AOT',      # Airports of Thailand
        'CPALL',    # CP All (7-Eleven Thailand)
    ]
    
    # Fetch stocks
    fetcher.fetch_multiple_stocks(major_stocks)
    
    # Fetch SET Index
    fetcher.fetch_set_index()
