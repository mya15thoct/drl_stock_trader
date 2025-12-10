"""
Quick US Stock Analyzer - Get 9 best US stocks
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def calc_metrics(df):
    """Calculate volatility and drawdown"""
    try:
        close = df['close_price'].values
        returns = np.diff(close) / close[:-1]
        
        vol = np.std(returns) * np.sqrt(252)
        
        rolling_max = np.maximum.accumulate(close)
        dd = np.abs(np.min((close / rolling_max - 1.0)))
        
        return {'vol': vol, 'dd': dd, 'days': len(df)}
    except:
        return None

# Analyze 50 US stocks
base_dir = Path(__file__).parent.parent / 'trainingset'
us_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
             'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD',
             'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA',
             'BA', 'CAT', 'GE', 'UPS', 'HON', 'MMM', 'LMT',
             'XOM', 'CVX', 'COP', 'SLB', 'EOG',
             'UNH', 'ABT', 'TMO', 'ABBV', 'MRK',
             'DIS', 'NFLX', 'CMCSA', 'T', 'VZ',
             'NKE', 'SBUX', 'HD', 'LOW', 'TGT', 'NEE']

results = []
for stock in us_stocks:
    csv_file = base_dir / f'{stock}.csv'
    if not csv_file.exists():
        continue
    
    df = pd.read_csv(csv_file)
    df = df[(df['date'] >= '2018-01-01') & (df['date'] < '2024-01-01')]
    
    if len(df) < 252:
        continue
    
    m = calc_metrics(df)
    if m:
        results.append({'stock': stock, **m})
        print(f"✓ {stock:6s} Vol={m['vol']:.3f} DD={m['dd']:.3f}")

# Convert to DataFrame and classify
df_results = pd.DataFrame(results)
print(f"\nTotal US stocks analyzed: {len(df_results)}")

# Calculate thresholds
vol_low = df_results['vol'].quantile(0.33)
vol_high = df_results['vol'].quantile(0.67)
dd_low = df_results['dd'].quantile(0.33)
dd_high = df_results['dd'].quantile(0.67)

print(f"\nUS THRESHOLDS:")
print(f"  Vol:  Low={vol_low:.3f}, High={vol_high:.3f}")
print(f"  DD:   Low={dd_low:.3f}, High={dd_high:.3f}")

# Classify
df_results['vol_score'] = (df_results['vol'] - vol_low) / (vol_high - vol_low)
df_results['dd_score'] = (df_results['dd'] - dd_low) / (dd_high - dd_low)
df_results['composite'] = df_results['vol_score'] * 0.6 + df_results['dd_score'] * 0.4

low_risk = df_results[df_results['composite'] < 0.33].sort_values('days', ascending=False).head(3)
med_risk = df_results[(df_results['composite'] >= 0.33) & (df_results['composite'] <= 0.67)].sort_values('days', ascending=False).head(3)
high_risk = df_results[df_results['composite'] > 0.67].sort_values('days', ascending=False).head(3)

print(f"\n{'='*60}")
print("9 RECOMMENDED US STOCKS FOR DRL TRAINING")
print(f"{'='*60}")

print("\nLow-Risk (3 stocks):")
for _, row in low_risk.iterrows():
    print(f"  • {row['stock']:6s} Vol={row['vol']:.3f} DD={row['dd']:.3f}")

print("\nMedium-Risk (3 stocks):")
for _, row in med_risk.iterrows():
    print(f"  • {row['stock']:6s} Vol={row['vol']:.3f} DD={row['dd']:.3f}")

print("\nHigh-Risk (3 stocks):")
for _, row in high_risk.iterrows():
    print(f"  • {row['stock']:6s} Vol={row['vol']:.3f} DD={row['dd']:.3f}")

# Save recommendation
us_9 = list(low_risk['stock']) + list(med_risk['stock']) + list(high_risk['stock'])
print(f"\nFINAL 9 US STOCKS: {', '.join(us_9)}")

with open('../classification/us_9_stocks.txt', 'w') as f:
    f.write('\n'.join(us_9))
print("Saved to classification/us_9_stocks.txt")
