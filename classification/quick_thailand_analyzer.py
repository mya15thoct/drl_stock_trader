"""
Quick analyzer for Thailand stocks
Analyzes stock performance and selects promising stocks for DRL trading
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class ThailandStockAnalyzer:
    """Analyzer for Thailand stock market data"""
    
    def __init__(self, data_dir='thailand_dataset'):
        self.data_dir = data_dir
        self.stocks_data = {}
        self.analysis_results = {}
    
    def load_stock(self, symbol):
        """Load stock data from CSV"""
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            return None
        
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            self.stocks_data[symbol] = df
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
            return None
    
    def calculate_metrics(self, symbol):
        """Calculate performance metrics for a stock"""
        
        if symbol not in self.stocks_data:
            df = self.load_stock(symbol)
            if df is None:
                return None
        else:
            df = self.stocks_data[symbol]
        
        if len(df) < 2:
            return None
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Basic metrics
        metrics = {
            'symbol': symbol,
            'start_date': df['Date'].min(),
            'end_date': df['Date'].max(),
            'days': len(df),
            'start_price': df['Close'].iloc[0],
            'end_price': df['Close'].iloc[-1],
            'total_return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100,
            'avg_volume': df['Volume'].mean(),
            'volatility': df['Returns'].std() * np.sqrt(252) * 100,  # Annualized
            'sharpe_ratio': (df['Returns'].mean() / df['Returns'].std()) * np.sqrt(252) if df['Returns'].std() > 0 else 0,
        }
        
        # Recent performance (last 3 months)
        recent_df = df[df['Date'] >= (df['Date'].max() - timedelta(days=90))]
        if len(recent_df) > 1:
            metrics['recent_return'] = (recent_df['Close'].iloc[-1] / recent_df['Close'].iloc[0] - 1) * 100
        else:
            metrics['recent_return'] = 0
        
        # Trend analysis
        if len(df) >= 50:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            if len(df) >= 200:
                metrics['trend'] = 'bullish' if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] else 'bearish'
            else:
                metrics['trend'] = 'neutral'
        else:
            metrics['trend'] = 'neutral'
        
        self.analysis_results[symbol] = metrics
        return metrics
    
    def analyze_all_stocks(self, symbols=None):
        """Analyze all stocks in the dataset"""
        
        if symbols is None:
            # Get all CSV files in the directory
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            symbols = [f.replace('.csv', '') for f in files if f != 'SET.csv']
        
        print(f"Analyzing {len(symbols)} Thailand stocks...")
        
        results = []
        for symbol in symbols:
            metrics = self.calculate_metrics(symbol)
            if metrics is not None:
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def select_top_stocks(self, n=10, criteria='sharpe_ratio'):
        """
        Select top N stocks based on specified criteria
        
        Args:
            n: Number of stocks to select
            criteria: Selection criteria ('sharpe_ratio', 'total_return', 'recent_return')
        
        Returns:
            List of top stock symbols
        """
        
        if not self.analysis_results:
            self.analyze_all_stocks()
        
        df = pd.DataFrame(self.analysis_results.values())
        
        # Filter out stocks with insufficient data
        df = df[df['days'] >= 200]  # At least 200 days of data
        df = df[df['avg_volume'] > 100000]  # Minimum volume
        
        # Sort by criteria
        df = df.sort_values(criteria, ascending=False)
        
        top_df = df.head(n)
        
        print(f"\nTop {n} Thailand stocks by {criteria}:")
        print("="*80)
        for i, row in top_df.iterrows():
            print(f"{row['symbol']:8s} | Return: {row['total_return']:6.2f}% | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f} | Vol: {row['volatility']:5.2f}% | "
                  f"Trend: {row['trend']}")
        
        top_stocks = top_df['symbol'].tolist()
        return top_stocks
    
    def save_selected_stocks(self, symbols, output_file='thailand_stocks.txt'):
        """Save selected stock symbols to a text file"""
        
        output_path = os.path.join('classification', output_file)
        
        with open(output_path, 'w') as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        
        print(f"\nSaved {len(symbols)} stocks to {output_path}")
    
    def generate_report(self):
        """Generate analysis report"""
        
        if not self.analysis_results:
            self.analyze_all_stocks()
        
        df = pd.DataFrame(self.analysis_results.values())
        
        print("\n" + "="*80)
        print("THAILAND STOCK MARKET ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nTotal stocks analyzed: {len(df)}")
        print(f"Date range: {df['start_date'].min()} to {df['end_date'].max()}")
        
        print("\nMarket Statistics:")
        print(f"  Average return: {df['total_return'].mean():.2f}%")
        print(f"  Average volatility: {df['volatility'].mean():.2f}%")
        print(f"  Average Sharpe ratio: {df['sharpe_ratio'].mean():.2f}")
        
        print("\nTop performers by total return:")
        top_return = df.nlargest(5, 'total_return')
        for _, row in top_return.iterrows():
            print(f"  {row['symbol']:8s}: {row['total_return']:6.2f}%")
        
        print("\nBest risk-adjusted returns (Sharpe ratio):")
        top_sharpe = df.nlargest(5, 'sharpe_ratio')
        for _, row in top_sharpe.iterrows():
            print(f"  {row['symbol']:8s}: {row['sharpe_ratio']:5.2f}")
        
        print("\nMost volatile stocks:")
        top_vol = df.nlargest(5, 'volatility')
        for _, row in top_vol.iterrows():
            print(f"  {row['symbol']:8s}: {row['volatility']:5.2f}%")
        
        return df


def main():
    """Main function to run quick analysis"""
    
    analyzer = ThailandStockAnalyzer()
    
    # Generate full report
    report_df = analyzer.generate_report()
    
    # Select top stocks
    print("\n" + "="*80)
    print("SELECTING TOP STOCKS FOR DRL TRADING")
    print("="*80)
    
    # Select top 9 stocks by Sharpe ratio (risk-adjusted return)
    top_stocks = analyzer.select_top_stocks(n=9, criteria='sharpe_ratio')
    
    # Save to file
    analyzer.save_selected_stocks(top_stocks, 'thailand_9_stocks.txt')
    
    # Save full report to CSV
    report_df.to_csv('classification/thailand_analysis_report.csv', index=False)
    print("\nSaved full analysis report to classification/thailand_analysis_report.csv")


if __name__ == "__main__":
    main()
