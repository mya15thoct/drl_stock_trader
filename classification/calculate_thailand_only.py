"""
Calculate Thailand thresholds ONLY
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.calculate_thresholds import calculate_stock_metrics, calculate_hurst_exponent

# Analyze Thailand market
print("="*70)
print("ANALYZING THAILAND MARKET - CALCULATING THRESHOLDS")
print("="*70)

data_path = Path('../dataset/thailand_dataset')
csv_files = list(data_path.glob('*.csv'))
csv_files = [f for f in csv_files if f.stem != 'SET']

print(f"Found {len(csv_files)} stock files")
print("Calculating 4 metrics for each stock...")
print("-"*70)

results = []
for i, csv_file in enumerate(csv_files, 1):
    symbol = csv_file.stem
    if i % 20 == 0:
        print(f"[{i}/{len(csv_files)}] Processing {symbol}...")
    
    metrics = calculate_stock_metrics(csv_file, market='Thailand')
    if metrics is not None:
        results.append({'symbol': symbol, **metrics})

print(f"\n{'='*70}")
print(f"Stocks with valid data: {len(results)}/{len(csv_files)}")

if len(results) < 100:
    print(f"WARNING: Only {len(results)} valid stocks. Need ~300 for reliable thresholds!")
    print("="*70)
else:
    df_results = pd.DataFrame(results)
    
    # Normalize and calculate composite score
    def normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin == 0:
            return pd.Series([0.0] * len(x))
        return (x - xmin) / (xmax - xmin)
    
    df_results['norm_vol'] = normalize(df_results['volatility'])
    df_results['norm_dd'] = normalize(df_results['max_drawdown'])
    df_results['norm_autocorr'] = normalize(df_results['autocorr'])
    df_results['norm_hurst'] = normalize(df_results['hurst'])
    
    df_results['composite_score'] = (
        0.40 * df_results['norm_vol'] +
        0.30 * df_results['norm_dd'] +
        0.20 * df_results['norm_autocorr'] +
        0.10 * df_results['norm_hurst']
    )
    
    # Calculate thresholds
    threshold_low = df_results['composite_score'].quantile(0.33)
    threshold_high = df_results['composite_score'].quantile(0.67)
    
    df_results['risk_class'] = pd.cut(
        df_results['composite_score'],
        bins=[-np.inf, threshold_low, threshold_high, np.inf],
        labels=['Low', 'Medium', 'High']
    )
    
    class_counts = df_results['risk_class'].value_counts()
    
    print(f"\n{'='*70}")
    print(f"THAILAND MARKET THRESHOLDS (COMPOSITE SCORE)")
    print(f"{'='*70}")
    print(f"  Low-Risk:    S < {threshold_low:.3f}  ({class_counts.get('Low', 0)} stocks, {class_counts.get('Low', 0)/len(df_results)*100:.1f}%)")
    print(f"  Medium-Risk: {threshold_low:.3f} ≤ S < {threshold_high:.3f}  ({class_counts.get('Medium', 0)} stocks, {class_counts.get('Medium', 0)/len(df_results)*100:.1f}%)")
    print(f"  High-Risk:   S ≥ {threshold_high:.3f}  ({class_counts.get('High', 0)} stocks, {class_counts.get('High', 0)/len(df_results)*100:.1f}%)")
    
    print(f"\nTHAILAND MEAN CHARACTERISTICS BY RISK CLASS:")
    print(f"{'='*70}")
    print(f"{'Class':<12} {'Volatility':<12} {'Max DD':<12} {'Autocorr':<12} {'Hurst':<12}")
    print("-"*70)
    
    for risk_class in ['Low', 'Medium', 'High']:
        class_data = df_results[df_results['risk_class'] == risk_class]
        if len(class_data) > 0:
            print(f"{risk_class:<12} {class_data['volatility'].mean():<12.3f} "
                  f"{class_data['max_drawdown'].mean():<12.3f} "
                  f"{class_data['autocorr'].mean():<12.3f} "
                  f"{class_data['hurst'].mean():<12.3f}")
    
    print("="*70)
    
    # Save results
    output_file = 'thailand_threshold_analysis.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved detailed analysis to: {output_file}")
    print("="*70)
