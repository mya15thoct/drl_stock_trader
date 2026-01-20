"""
Calculate Classification Thresholds for US and Thailand Markets
Following Vietnam methodology from PDF:
- Volatility (annualized)
- Maximum Drawdown
- Autocorrelation (1-day)
- Hurst Exponent
- Composite Score = 0.40*Vol + 0.30*DD + 0.20*Autocorr + 0.10*Hurst
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def calculate_hurst_exponent(prices):
    """
    Calculate Hurst exponent using R/S analysis
    H > 0.5: trending behavior (momentum)
    H < 0.5: mean-reverting behavior
    H = 0.5: random walk
    """
    if len(prices) < 100:
        return 0.0

    try:
        # Use log prices
        log_prices = np.log(prices)

        # Calculate different time lags
        lags = range(2, min(100, len(log_prices)//2))
        tau = []
        rs_values = []

        for lag in lags:
            # Divide series into subseries of length lag
            n = len(log_prices) // lag
            if n == 0:
                continue

            subseries = [log_prices[i*lag:(i+1)*lag] for i in range(n)]

            # Calculate R/S for each subseries
            rs_list = []
            for sub in subseries:
                if len(sub) < 2:
                    continue

                mean_sub = np.mean(sub)
                deviations = sub - mean_sub
                cumsum_dev = np.cumsum(deviations)

                R = np.max(cumsum_dev) - np.min(cumsum_dev)
                S = np.std(sub, ddof=1)

                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                tau.append(lag)
                rs_values.append(np.mean(rs_list))

        if len(tau) < 2:
            return 0.0

        # Hurst exponent from slope of log(R/S) vs log(lag)
        log_tau = np.log(tau)
        log_rs = np.log(rs_values)

        # Linear regression
        coeffs = np.polyfit(log_tau, log_rs, 1)
        hurst = coeffs[0]

        return hurst

    except Exception as e:
        print(f"  Warning: Hurst calculation failed: {e}")
        return 0.0


def calculate_stock_metrics(csv_path, market='US'):
    """
    Calculate all 4 metrics for one stock

    Args:
        csv_path: Path to CSV file
        market: 'US' or 'Thailand'

    Returns:
        Dict with metrics or None
    """
    try:
        df = pd.read_csv(csv_path)

        # Handle different column names
        if market == 'US':
            # US format: date, close_price, open_price, high_price, low_price, volume_detail
            if 'close_price' in df.columns:
                close_col = 'close_price'
            elif 'Close' in df.columns:
                close_col = 'Close'
            else:
                return None
        else:
            # Thailand format: Date, Close, Open, High, Low, Volume
            if 'Close' in df.columns:
                close_col = 'Close'
            elif 'close_price' in df.columns:
                close_col = 'close_price'
            else:
                return None

        # Filter data: 2018-2023 (training period like Vietnam)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            start_date = pd.to_datetime('2018-01-01', utc=True)
            end_date = pd.to_datetime('2024-01-01', utc=True)
            df = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            start_date = pd.to_datetime('2018-01-01', utc=True)
            end_date = pd.to_datetime('2024-01-01', utc=True)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]

        # Need minimum 252 trading days (1 year)
        if len(df) < 252:
            return None

        # Get close prices
        close_prices = df[close_col].values

        # Remove NaN
        close_prices = close_prices[~np.isnan(close_prices)]

        if len(close_prices) < 252:
            return None

        # 1. VOLATILITY (annualized)
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            return None

        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # 2. MAXIMUM DRAWDOWN
        rolling_max = np.maximum.accumulate(close_prices)
        drawdown = (close_prices / rolling_max - 1.0)
        max_drawdown = np.abs(np.min(drawdown))

        # 3. AUTOCORRELATION (1-day lag)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        # 4. HURST EXPONENT
        hurst = calculate_hurst_exponent(close_prices)

        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'autocorr': autocorr,
            'hurst': hurst,
            'days': len(df)
        }

    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return None


def analyze_market(data_dir, market_name):
    """
    Analyze all stocks in a market and calculate thresholds

    Args:
        data_dir: Directory containing CSV files
        market_name: 'US' or 'Thailand'

    Returns:
        DataFrame with results and threshold values
    """
    print("="*70)
    print(f"ANALYZING {market_name} MARKET - CALCULATING THRESHOLDS")
    print("="*70)

    # Get all CSV files
    data_path = Path(data_dir)
    csv_files = list(data_path.glob('*.csv'))

    # Exclude index files
    if market_name == 'US':
        csv_files = [f for f in csv_files if f.stem not in ['GSPC', 'INDEX_GSPC', 'INDEX_DJI', 'INDEX_IXIC']]
    else:
        csv_files = [f for f in csv_files if f.stem != 'SET']

    print(f"Found {len(csv_files)} stock files")
    print(f"Calculating 4 metrics for each stock...")
    print("-"*70)

    results = []

    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.stem

        if i % 20 == 0:
            print(f"[{i}/{len(csv_files)}] Processing {symbol}...")

        metrics = calculate_stock_metrics(csv_file, market=market_name)

        if metrics is not None:
            results.append({
                'symbol': symbol,
                **metrics
            })

    print(f"\n{'='*70}")
    print(f"Stocks with valid data: {len(results)}/{len(csv_files)}")

    if len(results) < 100:
        print(f"WARNING: Only {len(results)} valid stocks. Need ~300 for reliable thresholds!")
        print("="*70)
        return None, None

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # CALCULATE COMPOSITE SCORE (same formula as Vietnam)
    # Normalize each metric using min-max scaling
    def normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin == 0:
            return pd.Series([0.0] * len(x))
        return (x - xmin) / (xmax - xmin)

    df_results['norm_vol'] = normalize(df_results['volatility'])
    df_results['norm_dd'] = normalize(df_results['max_drawdown'])
    df_results['norm_autocorr'] = normalize(df_results['autocorr'])
    df_results['norm_hurst'] = normalize(df_results['hurst'])

    # Composite score: S = 0.40*Vol + 0.30*DD + 0.20*Autocorr + 0.10*Hurst
    df_results['composite_score'] = (
        0.40 * df_results['norm_vol'] +
        0.30 * df_results['norm_dd'] +
        0.20 * df_results['norm_autocorr'] +
        0.10 * df_results['norm_hurst']
    )

    # CALCULATE THRESHOLDS (tertiles)
    threshold_low = df_results['composite_score'].quantile(0.33)
    threshold_high = df_results['composite_score'].quantile(0.67)

    # Classify stocks
    df_results['risk_class'] = pd.cut(
        df_results['composite_score'],
        bins=[-np.inf, threshold_low, threshold_high, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    # Count by class
    class_counts = df_results['risk_class'].value_counts()

    print(f"\n{'='*70}")
    print(f"{market_name} MARKET THRESHOLDS (COMPOSITE SCORE)")
    print(f"{'='*70}")
    print(f"  Low-Risk:    S < {threshold_low:.3f}  ({class_counts.get('Low', 0)} stocks, {class_counts.get('Low', 0)/len(df_results)*100:.1f}%)")
    print(f"  Medium-Risk: {threshold_low:.3f} ≤ S < {threshold_high:.3f}  ({class_counts.get('Medium', 0)} stocks, {class_counts.get('Medium', 0)/len(df_results)*100:.1f}%)")
    print(f"  High-Risk:   S ≥ {threshold_high:.3f}  ({class_counts.get('High', 0)} stocks, {class_counts.get('High', 0)/len(df_results)*100:.1f}%)")

    # Show mean values per class
    print(f"\n{market_name} MEAN CHARACTERISTICS BY RISK CLASS:")
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
    output_file = f'{market_name.lower()}_threshold_analysis.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved detailed analysis to: {output_file}")

    thresholds = {
        'threshold_low': threshold_low,
        'threshold_high': threshold_high,
        'total_stocks': len(df_results),
        'low_count': class_counts.get('Low', 0),
        'medium_count': class_counts.get('Medium', 0),
        'high_count': class_counts.get('High', 0)
    }

    return df_results, thresholds


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STOCK CLASSIFICATION THRESHOLD CALCULATOR")
    print("Following Vietnam Methodology (PDF)")
    print("="*70)
    print("\nMetrics:")
    print("  1. Volatility (annualized)")
    print("  2. Maximum Drawdown")
    print("  3. Autocorrelation (1-day lag)")
    print("  4. Hurst Exponent (R/S analysis)")
    print("\nComposite Score: S = 0.40*Vol + 0.30*DD + 0.20*Autocorr + 0.10*Hurst")
    print("="*70)

    # Analyze US market
    print("\n")
    us_results, us_thresholds = analyze_market('dataset/us_dataset', 'US')

    # Analyze Thailand market
    print("\n\n")
    th_results, th_thresholds = analyze_market('thailand_dataset', 'Thailand')

    # Summary comparison
    print("\n" + "="*70)
    print("MARKET COMPARISON SUMMARY")
    print("="*70)

    if us_thresholds:
        print(f"\nUS MARKET:")
        print(f"  Total stocks analyzed: {us_thresholds['total_stocks']}")
        print(f"  Threshold Low:  {us_thresholds['threshold_low']:.3f}")
        print(f"  Threshold High: {us_thresholds['threshold_high']:.3f}")

    if th_thresholds:
        print(f"\nTHAILAND MARKET:")
        print(f"  Total stocks analyzed: {th_thresholds['total_stocks']}")
        print(f"  Threshold Low:  {th_thresholds['threshold_low']:.3f}")
        print(f"  Threshold High: {th_thresholds['threshold_high']:.3f}")

    print("\n" + "="*70)
    print("DONE! Thresholds calculated for both markets")
    print("="*70)
