"""
VN MARKET THRESHOLD CALCULATOR
Tính toán ngưỡng phân loại từ 256 mã VN-Index để tạo benchmark cho thị trường VN

Usage:
    python vn_threshold_calculator.py

Input:
    - Thư mục chứa dữ liệu 256 mã VN-Index
    
Output:
    - File JSON chứa ngưỡng cho classification
    - Báo cáo thống kê distribution
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class VNMarketThresholdCalculator:
    """Tính toán ngưỡng từ toàn bộ thị trường VN"""
    
    def __init__(self, vnindex_data_directory):
        self.data_directory = vnindex_data_directory
        self.market_metrics = {}
        self.thresholds = {}
        
    def calculate_market_thresholds(self):
        """Tính toán ngưỡng từ toàn bộ thị trường VN"""
        
        print("=== VN MARKET THRESHOLD CALCULATION ===")
        print(f"Data directory: {self.data_directory}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Lấy tất cả file CSV
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {self.data_directory}")
            return
        
        print(f"Found {len(csv_files)} stocks to analyze...")
        print()
        
        # Thu thập metrics từ tất cả cổ phiếu
        all_metrics = []
        successful_count = 0
        
        for i, csv_file in enumerate(csv_files, 1):
            stock_code = csv_file.replace('.csv', '')
            file_path = os.path.join(self.data_directory, csv_file)
            
            if i % 50 == 0:
                print(f"Progress: {i}/{len(csv_files)} stocks processed...")
            
            try:
                metrics = self._calculate_stock_metrics(file_path, stock_code)
                if metrics:
                    all_metrics.append(metrics)
                    successful_count += 1
                    
            except Exception as e:
                print(f"Error processing {stock_code}: {str(e)}")
                continue
        
        print(f"\nSuccessfully processed: {successful_count}/{len(csv_files)} stocks")
        
        if successful_count < 100:
            print("WARNING: Too few stocks processed for reliable thresholds")
            return
        
        # Tính toán ngưỡng từ distribution
        self._compute_thresholds(all_metrics)
        
        # Tạo báo cáo và xuất file
        self._generate_report()
        self._export_thresholds()
        
        return self.thresholds
    
    def _calculate_stock_metrics(self, file_path, stock_code):
        """Tính toán metrics cho một cổ phiếu"""
        
        try:
            # Load và xử lý dữ liệu
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'date': 'Date',
                'close_price': 'Close',
                'open_price': 'Open', 
                'high_price': 'High',
                'low_price': 'Low',
                'volume_detail': 'Volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Date processing
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df[(df['Date'] >= '2018-01-01') & (df['Date'] < '2025-01-01')]
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Numeric conversion
            for col in ['Close', 'Open', 'High', 'Low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Volume handling
            if 'Volume' not in df.columns and 'volume_detail' in df.columns:
                df['Volume'] = pd.to_numeric(df['volume_detail'], errors='coerce')
            elif 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            else:
                df['Volume'] = 1000000
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df = df.dropna()
            
            # Minimum data requirement
            if len(df) < 252:  # Less than 1 year
                return None
            
            # Calculate core metrics
            returns = df['Returns'].dropna()
            close_prices = df['Close']
            
            # VOLATILITY METRICS
            daily_vol = returns.std()
            annualized_volatility = daily_vol * np.sqrt(252)
            
            # Downside volatility
            negative_returns = returns[returns < 0]
            downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # RISK METRICS
            rolling_max = close_prices.expanding().max()
            drawdown = (close_prices / rolling_max - 1.0)
            max_drawdown = drawdown.min()
            
            # VaR
            var_5 = returns.quantile(0.05)
            
            # MOMENTUM/MEAN REVERSION
            autocorr_1d = returns.autocorr(lag=1) if len(returns) > 1 else 0
            autocorr_5d = returns.autocorr(lag=5) if len(returns) > 5 else 0
            
            # Hurst exponent
            def hurst_simplified(returns_series):
                try:
                    lags = [1, 2, 4, 8, 16]
                    variances = [returns_series.diff(lag).var() for lag in lags]
                    variances = [v for v in variances if not np.isnan(v) and v > 0]
                    if len(variances) < 3:
                        return 0.5
                    log_lags = np.log(lags[:len(variances)])
                    log_vars = np.log(variances)
                    slope = np.polyfit(log_lags, log_vars, 1)[0]
                    return slope / 2
                except:
                    return 0.5
            
            hurst_exponent = hurst_simplified(returns)
            
            # TREND CHARACTERISTICS
            ma_20 = close_prices.rolling(20, min_periods=1).mean()
            ma_50 = close_prices.rolling(50, min_periods=1).mean()
            trend_signal = (ma_20 > ma_50).astype(int)
            trend_changes = trend_signal.diff().abs().sum()
            trend_change_frequency = trend_changes / len(df) if len(df) > 0 else 0
            
            # SKEWNESS AND KURTOSIS
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            return {
                'stock_code': stock_code,
                'annualized_volatility': annualized_volatility,
                'downside_volatility': downside_volatility,
                'max_drawdown': abs(max_drawdown),
                'var_5': abs(var_5),
                'autocorr_1d': abs(autocorr_1d),
                'autocorr_5d': abs(autocorr_5d),
                'hurst_exponent': hurst_exponent,
                'trend_change_frequency': trend_change_frequency,
                'skewness': abs(skewness),
                'kurtosis': abs(kurtosis),
                'sample_size': len(df)
            }
            
        except Exception as e:
            print(f"Error calculating metrics for {stock_code}: {str(e)}")
            return None
    
    def _compute_thresholds(self, all_metrics):
        """Tính toán ngưỡng từ distribution của tất cả cổ phiếu"""
        
        # Extract arrays for each metric
        metrics_arrays = {}
        for metric_name in ['annualized_volatility', 'max_drawdown', 'autocorr_1d', 
                           'hurst_exponent', 'trend_change_frequency', 'skewness', 'kurtosis']:
            values = [stock[metric_name] for stock in all_metrics if not np.isnan(stock[metric_name])]
            metrics_arrays[metric_name] = np.array(values)
        
        # Store market distribution
        self.market_metrics = metrics_arrays
        
        # Calculate thresholds using percentiles
        self.thresholds = {}
        
        # PRIMARY CLASSIFICATION THRESHOLDS
        vol_array = metrics_arrays['annualized_volatility']
        
        # Sử dụng terciles (33%, 67%) để chia 3 nhóm đều
        vol_33rd = np.percentile(vol_array, 33.33)
        vol_67th = np.percentile(vol_array, 66.67)
        
        self.thresholds['volatility'] = {
            'low_threshold': vol_33rd,
            'high_threshold': vol_67th,
            'description': 'Low: <33rd percentile, Medium: 33rd-67th, High: >67th percentile'
        }
        
        # RISK THRESHOLDS
        drawdown_array = metrics_arrays['max_drawdown']
        self.thresholds['risk'] = {
            'low_threshold': np.percentile(drawdown_array, 33.33),
            'high_threshold': np.percentile(drawdown_array, 66.67)
        }
        
        # MOMENTUM THRESHOLDS
        momentum_array = metrics_arrays['autocorr_1d']
        self.thresholds['momentum'] = {
            'low_threshold': np.percentile(momentum_array, 25),    # 25% có momentum thấp nhất
            'high_threshold': np.percentile(momentum_array, 75)    # 25% có momentum cao nhất
        }
        
        # MEAN REVERSION THRESHOLDS (Hurst)
        hurst_array = metrics_arrays['hurst_exponent']
        self.thresholds['mean_reversion'] = {
            'strong_reversion_threshold': np.percentile(hurst_array, 25),  # Hurst < 25th = strong mean reversion
            'momentum_trending_threshold': np.percentile(hurst_array, 75)  # Hurst > 75th = momentum trending
        }
        
        # TREND FREQUENCY THRESHOLDS
        trend_freq_array = metrics_arrays['trend_change_frequency']
        self.thresholds['trend_frequency'] = {
            'low_turnover_threshold': np.percentile(trend_freq_array, 33.33),
            'high_turnover_threshold': np.percentile(trend_freq_array, 66.67)
        }
        
        # ADDITIONAL METADATA
        self.thresholds['metadata'] = {
            'total_stocks_analyzed': len(all_metrics),
            'calculation_date': datetime.now().isoformat(),
            'data_period': '2018-2025',
            'market': 'VN_INDEX',
            'methodology': 'percentile_based_terciles'
        }
        
        # NORMALIZATION RANGES (for 0-1 scaling)
        self.thresholds['normalization_ranges'] = {
            'volatility': {
                'min_val': np.percentile(vol_array, 5),      # 5th percentile
                'max_val': np.percentile(vol_array, 95)      # 95th percentile
            },
            'risk': {
                'min_val': np.percentile(drawdown_array, 5),
                'max_val': np.percentile(drawdown_array, 95)
            }
        }
    
    def _generate_report(self):
        """Tạo báo cáo thống kê"""
        
        print("\n=== VN MARKET THRESHOLD REPORT ===")
        print(f"Total stocks analyzed: {self.thresholds['metadata']['total_stocks_analyzed']}")
        print()
        
        # Volatility thresholds
        vol_thresholds = self.thresholds['volatility']
        print("VOLATILITY CLASSIFICATION THRESHOLDS:")
        print(f"  Low Volatility:    < {vol_thresholds['low_threshold']:.4f} ({vol_thresholds['low_threshold']*100:.2f}%)")
        print(f"  Medium Volatility: {vol_thresholds['low_threshold']:.4f} - {vol_thresholds['high_threshold']:.4f}")
        print(f"  High Volatility:   > {vol_thresholds['high_threshold']:.4f} ({vol_thresholds['high_threshold']*100:.2f}%)")
        print()
        
        # Risk thresholds
        risk_thresholds = self.thresholds['risk']
        print("RISK CLASSIFICATION THRESHOLDS:")
        print(f"  Low Risk:     < {risk_thresholds['low_threshold']:.4f}")
        print(f"  Medium Risk:  {risk_thresholds['low_threshold']:.4f} - {risk_thresholds['high_threshold']:.4f}")
        print(f"  High Risk:    > {risk_thresholds['high_threshold']:.4f}")
        print()
        
        # Momentum thresholds
        momentum_thresholds = self.thresholds['momentum']
        print("MOMENTUM THRESHOLDS:")
        print(f"  Low Momentum:  < {momentum_thresholds['low_threshold']:.4f}")
        print(f"  High Momentum: > {momentum_thresholds['high_threshold']:.4f}")
        print()
        
        # Distribution statistics
        vol_array = self.market_metrics['annualized_volatility']
        print("MARKET VOLATILITY DISTRIBUTION:")
        print(f"  Min:    {np.min(vol_array):.4f}")
        print(f"  Q1:     {np.percentile(vol_array, 25):.4f}")
        print(f"  Median: {np.median(vol_array):.4f}")
        print(f"  Q3:     {np.percentile(vol_array, 75):.4f}")
        print(f"  Max:    {np.max(vol_array):.4f}")
        print(f"  Mean:   {np.mean(vol_array):.4f}")
        print(f"  Std:    {np.std(vol_array):.4f}")
    
    def _export_thresholds(self):
        """Xuất ngưỡng ra file JSON"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'vn_market_thresholds_{timestamp}.json'
        
        # Add distribution stats to export
        export_data = {
            'thresholds': self.thresholds,
            'market_distribution_stats': {}
        }
        
        # Add distribution statistics for each metric
        for metric_name, values in self.market_metrics.items():
            export_data['market_distribution_stats'][metric_name] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'q25': float(np.percentile(values, 25)),
                'median': float(np.median(values)),
                'q75': float(np.percentile(values, 75)),
                'max': float(np.max(values)),
                'percentiles': {
                    '5th': float(np.percentile(values, 5)),
                    '10th': float(np.percentile(values, 10)),
                    '33rd': float(np.percentile(values, 33.33)),
                    '67th': float(np.percentile(values, 66.67)),
                    '90th': float(np.percentile(values, 90)),
                    '95th': float(np.percentile(values, 95))
                }
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nThresholds exported to: {filename}")
        print("This file can be used in your classification system.")
        
        return filename
    
    def plot_distributions(self):
        """Vẽ biểu đồ distribution (optional)"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('VN Market Metrics Distribution', fontsize=16)
            
            # Volatility distribution
            vol_data = self.market_metrics['annualized_volatility']
            axes[0,0].hist(vol_data, bins=50, alpha=0.7, edgecolor='black')
            axes[0,0].axvline(self.thresholds['volatility']['low_threshold'], color='red', linestyle='--', label='33rd percentile')
            axes[0,0].axvline(self.thresholds['volatility']['high_threshold'], color='red', linestyle='--', label='67th percentile')
            axes[0,0].set_title('Annualized Volatility Distribution')
            axes[0,0].set_xlabel('Volatility')
            axes[0,0].legend()
            
            # Risk distribution
            risk_data = self.market_metrics['max_drawdown']
            axes[0,1].hist(risk_data, bins=50, alpha=0.7, edgecolor='black')
            axes[0,1].axvline(self.thresholds['risk']['low_threshold'], color='red', linestyle='--')
            axes[0,1].axvline(self.thresholds['risk']['high_threshold'], color='red', linestyle='--')
            axes[0,1].set_title('Max Drawdown Distribution')
            axes[0,1].set_xlabel('Max Drawdown')
            
            # Momentum distribution
            momentum_data = self.market_metrics['autocorr_1d']
            axes[1,0].hist(momentum_data, bins=50, alpha=0.7, edgecolor='black')
            axes[1,0].axvline(self.thresholds['momentum']['low_threshold'], color='red', linestyle='--')
            axes[1,0].axvline(self.thresholds['momentum']['high_threshold'], color='red', linestyle='--')
            axes[1,0].set_title('Momentum (Autocorr 1d) Distribution')
            axes[1,0].set_xlabel('Autocorrelation')
            
            # Hurst distribution
            hurst_data = self.market_metrics['hurst_exponent']
            axes[1,1].hist(hurst_data, bins=50, alpha=0.7, edgecolor='black')
            axes[1,1].axvline(self.thresholds['mean_reversion']['strong_reversion_threshold'], color='red', linestyle='--')
            axes[1,1].axvline(self.thresholds['mean_reversion']['momentum_trending_threshold'], color='red', linestyle='--')
            axes[1,1].set_title('Hurst Exponent Distribution')
            axes[1,1].set_xlabel('Hurst Exponent')
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'vn_market_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved as: vn_market_distribution_{timestamp}.png")
            
        except ImportError:
            print("Matplotlib not available for plotting. Skipping visualization.")

def main():
    """Main function"""
    
    # Đường dẫn thư mục chứa 256 mã VN-Index
    vnindex_directory = r"C:\IT\stock_lab.uit\drl - Copy\representative_stocks_data"  # Thay đổi path này
    
    # Kiểm tra thư mục
    if not os.path.exists(vnindex_directory):
        print(f"Directory not found: {vnindex_directory}")
        print("Please update the path to your VN-Index data directory.")
        return
    
    # Tính toán ngưỡng
    calculator = VNMarketThresholdCalculator(vnindex_directory)
    thresholds = calculator.calculate_market_thresholds()
    
    if thresholds:
        # Vẽ biểu đồ (optional)
        try:
            calculator.plot_distributions()
        except:
            print("Could not generate plots.")
        
        print("\n🎉 VN Market thresholds calculation completed!")
        print("Use the generated JSON file in your classification system.")
    else:
        print("❌ Failed to calculate thresholds.")

if __name__ == "__main__":
    main()