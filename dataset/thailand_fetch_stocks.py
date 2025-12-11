"""
Fetch Thailand stocks for DRL trading
Includes SET50 and additional major stocks
"""

from thailand_fetcher import ThailandStockFetcher
from datetime import datetime, timedelta
import time

# SET50 Index components (Top 50 largest stocks)
SET50_STOCKS = [
    # Energy
    'PTT', 'PTTEP', 'TOP', 'PTTGC', 'BANPU', 'RATCH',
    
    # Banking & Finance
    'KBANK', 'SCB', 'BBL', 'KTB', 'BAY', 'TISCO',
    
    # Commerce
    'CPALL', 'MAKRO', 'HMPRO', 'CRC', 'GLOBAL',
    
    # Property & Construction
    'AP', 'AWC', 'LH', 'CPN', 'SPALI',
    
    # Food & Beverage
    'CPF', 'MINT', 'OSP', 'CBG',
    
    # Technology & Communications
    'ADVANC', 'TRUE', 'INTUCH', 'DTAC',
    
    # Transportation & Logistics
    'AOT', 'BEM', 'BTS', 'TASCO',
    
    # Industrial
    'SCC', 'TPIPL', 'SCGP', 'IVL',
    
    # Healthcare
    'BCH', 'BH', 'BDMS',
    
    # Agro & Food Industry
    'TU', 'STA',
    
    # Media & Publishing
    'GRAMMY',
    
    # Tourism & Leisure
    'CENTEL',
    
    # Others
    'IRPC', 'WHA',
]

# Additional major stocks (SET100 and liquid stocks)
ADDITIONAL_STOCKS = [
    # More Banking & Finance
    'TCAP', 'TTB', 'SAWAD', 'MTC',
    
    # More Energy & Utilities
    'EGCO', 'GPSC', 'GULF', 'EA', 'BGRIM', 'BCP', 'ESSO',
    
    # More Property & Construction
    'ONYX', 'PSH', 'LPN', 'QH', 'SIRI', 'SC',
    
    # More Commerce & Retail
    'ROBINS', 'BJC', 'COM7', 'SINGER', 'MEGA',
    
    # Technology & Media
    'DELTA', 'KCE', 'HANA', 'WORK',
    
    # Industrial & Materials
    'TPIPP', 'GFPT', 'TTA', 'TSTE',
    
    # Healthcare & Pharma
    'CHG', 'RJH', 'PR9', 'RAM',
    
    # Food & Agriculture
    'SPC', 'GLOAT', 'NER',
    
    # Transportation
    'THAI', 'PSL',
]


def fetch_set50_stocks(delay=2):
    """Fetch SET50 stocks with delay to avoid rate limiting"""
    
    fetcher = ThailandStockFetcher(output_dir='thailand_dataset')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"Fetching {len(SET50_STOCKS)} SET50 stocks...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("="*60)
    
    # Fetch SET Index first
    print("\nFetching SET Index...")
    fetcher.fetch_set_index(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Fetch stocks with delay
    results = {}
    for i, symbol in enumerate(SET50_STOCKS, 1):
        print(f"\n[{i}/{len(SET50_STOCKS)}] {symbol}...")
        
        try:
            df = fetcher.fetch_stock(symbol, 
                                    start_date=start_date.strftime('%Y-%m-%d'),
                                    end_date=end_date.strftime('%Y-%m-%d'))
            if df is not None:
                base_symbol = symbol.replace('.BK', '')
                results[base_symbol] = df
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        if i < len(SET50_STOCKS) and delay > 0:
            time.sleep(delay)
    
    print("\n" + "="*60)
    print(f"Successfully fetched {len(results)}/{len(SET50_STOCKS)} stocks")
    
    return results


def fetch_additional_stocks(delay=2):
    """Fetch additional stocks beyond SET50"""
    
    fetcher = ThailandStockFetcher(output_dir='thailand_dataset')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\nFetching {len(ADDITIONAL_STOCKS)} additional stocks...")
    print("="*60)
    
    results = {}
    for i, symbol in enumerate(ADDITIONAL_STOCKS, 1):
        print(f"[{i}/{len(ADDITIONAL_STOCKS)}] {symbol}...")
        
        try:
            df = fetcher.fetch_stock(symbol,
                                    start_date=start_date.strftime('%Y-%m-%d'),
                                    end_date=end_date.strftime('%Y-%m-%d'))
            if df is not None:
                base_symbol = symbol.replace('.BK', '')
                results[base_symbol] = df
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        if i < len(ADDITIONAL_STOCKS) and delay > 0:
            time.sleep(delay)
    
    print("\n" + "="*60)
    print(f"Successfully fetched {len(results)}/{len(ADDITIONAL_STOCKS)} additional stocks")
    
    return results


def fetch_all_stocks(delay=2):
    """Fetch all Thailand stocks (SET50 + additional)"""
    
    print("="*60)
    print("FETCHING THAILAND STOCKS FOR DRL TRADING")
    print("="*60)
    
    # Fetch SET50
    set50_results = fetch_set50_stocks(delay=delay)
    
    # Fetch additional stocks
    additional_results = fetch_additional_stocks(delay=delay)
    
    # Summary
    total_stocks = len(set50_results) + len(additional_results)
    total_attempted = len(SET50_STOCKS) + len(ADDITIONAL_STOCKS)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SET50 stocks: {len(set50_results)}/{len(SET50_STOCKS)}")
    print(f"Additional stocks: {len(additional_results)}/{len(ADDITIONAL_STOCKS)}")
    print(f"Total stocks fetched: {total_stocks}/{total_attempted}")
    print("="*60)
    
    return {**set50_results, **additional_results}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch Thailand stocks')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['set50', 'additional', 'all'],
                       help='Which stocks to fetch')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    if args.mode == 'set50':
        fetch_set50_stocks(delay=args.delay)
    elif args.mode == 'additional':
        fetch_additional_stocks(delay=args.delay)
    else:
        fetch_all_stocks(delay=args.delay)
