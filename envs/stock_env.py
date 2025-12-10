'''

1. File này định nghĩa môi trường giao dịch cổ phiếu FPT cho Reinforcement Learning.
2. Hỗ trợ các chỉ báo kỹ thuật như SMA, Bollinger Bands, RSI, MACD để cung cấp thông tin thị trường.
3. Xử lý mua/bán cổ phiếu dựa trên hành động của agent, bao gồm tính toán phí giao dịch và giới hạn số lượng cổ phiếu.
4. Sử dụng phần thưởng dựa trên hiệu suất so với thị trường để khuyến khích chiến lược giao dịch tốt hơn.
5. Cung cấp các phương thức reset, step và kiểm thử môi trường với chiến lược ngẫu nhiên và "mua và giữ".
'''
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import random as rd
import warnings
warnings.filterwarnings('ignore') 
ARY = np.ndarray

class StockTradingEnv:
    def __init__(self, data_path, stock_code=None, initial_amount=1e6, max_stock=1e2,
                cost_pct=1e-3, gamma=0.99, tech_indicators_list=None, use_turbulence=False,
                train_test_split=None, use_train=True, predefined_stock_type=None):
        
        # Parameter setting
        self.data_path = data_path
        self.initial_amount = initial_amount
        self.max_stock = max_stock
        self.cost_pct = cost_pct
        self.reward_scale = 1.0
        self.gamma = gamma
        
        # Store train/test parameters
        self.train_test_split = train_test_split
        self.use_train = use_train

        # Xác định mã cổ phiếu từ tên file nếu không được cung cấp
        if stock_code is None:
            file_name = os.path.basename(data_path)
            stock_code = file_name.split('.')[0]
        self.stock_code = stock_code
        
        # Process data
        self.tech_indicators_list = tech_indicators_list if tech_indicators_list is not None else [
            "ma5", "ma10", "ma20", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"
        ]
        self.use_turbulence = use_turbulence

        # Load data ONCE
        self.load_data(data_path, train_test_split, use_train)
        
        # Classification logic
        if predefined_stock_type is not None:
            # Testing mode: use predefined classification
            self.stock_type = predefined_stock_type
            print(f"Using predefined stock type: {predefined_stock_type}")
        else:
            # Training mode: compute classification from data
            self.calculate_statistics()
            self.classify_stock()
        
        # Environment setup
        self.shares_num = 1
        self.day = None
        self.amount = None
        self.shares = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.if_random_reset = True

        # Environment info
        self.env_name = f'{self.stock_code}StockTradingEnv-v1'
        self.state_dim = 1 + 1 + self.close_ary.shape[1] + self.tech_ary.shape[1]
        self.action_dim = 1
        self.if_discrete = False
        self.max_step = len(self.close_ary) - 1
        self.target_return = +np.inf
    def load_data(self, data_path:str, train_test_split=None, use_train=True) -> None:
        """
        Đọc dữ liệu từ file CSV, phân chia dữ liệu và tính toán các chỉ báo kỹ thuật
        """
        # Đọc file csv
        df = pd.read_csv(data_path)
        
        # Map column names if they don't match expected format
        column_mapping = {
            'date': 'Date',
            'close_price': 'close_price',
            'open_price': 'open_price',
            'high_price': 'high_price',
            'low_price': 'low_price',
            'volume_detail': 'volume'
        }
        
        # Rename columns if needed
        df = df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns})
        
        # Ensure 'Date' column is in datetime format
        if 'Date' in df.columns:
            # Chuyển đổi cột Date sang datetime trước khi thực hiện bất kỳ phép toán nào
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Loại bỏ các hàng có ngày không hợp lệ
            
            # Hiện giờ in ra thông tin ngày sau khi đã chuyển đổi kiểu
            # print(f"Original date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Tiếp tục lọc dữ liệu theo khoảng thời gian
            # Handle both string and Timestamp comparison (pandas 2.0+ compatibility)
            try:
                df = df[(df['Date'] >= '2018-01-01') & (df['Date'] < '2025-01-01')]
            except TypeError:
                # If string comparison fails, convert to Timestamp
                df = df[(df['Date'] >= pd.Timestamp('2018-01-01')) & (df['Date'] < pd.Timestamp('2025-01-01'))]
            # print(f"After date conversion: {df['Date'].min()} to {df['Date'].max()}")
        
        # Sắp xếp dữ liệu theo ngày tăng dần (từ cũ đến mới)
        df = df.sort_values('Date', ascending=True)
        # print(f"After sorting: {df['Date'].min()} to {df['Date'].max()} - {len(df)} rows")
        
        # Phân chia data become train data and test data
        if train_test_split is not None:
            split_date = pd.Timestamp('2024-01-01')
            # Lưu lại số lượng dữ liệu ban đầu
            total_rows = len(df)
            train_data = df[df['Date'] < split_date]
            test_data = df[df['Date'] >= split_date]
            
     
            if use_train:
                df = train_data
                # print(f"Using training data: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
            else:
                df = test_data
                # print(f"Using testing data: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
        
        # Save in4 date:
        self.dates = df['Date'].values
        # Chuyển đổi các cột số liệu sang kiểu số
        for col in ['close_price', 'open_price', 'high_price', 'low_price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Chuyển đổi cột volume nếu có
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        if 'volume_detail' in df.columns:
            df['volume_detail'] = pd.to_numeric(df['volume_detail'], errors='coerce')


        # Tính toán các chỉ báo kỹ thuật
        df = self.add_technical_indicators(df)
        

        # Tạo array cho giá đóng cửa
        self.close_ary = df[['close_price']].values
        
        # Tạo array cho các chỉ báo kỹ thuật
        tech_list = []
        for indicator in self.tech_indicators_list:
            if indicator in df.columns:
                tech_list.append(df[indicator].values)
            else:
                print(f"Warning: Indicator '{indicator}' not found in DataFrame columns")
        
        # Ghép các chỉ báo kỹ thuật thành một mảng
        self.tech_ary = np.column_stack(tech_list) if tech_list else np.array([])
        
    
    def calculate_statistics(self):
        """Tính toán các thông số thống kê cơ bản"""
        if not hasattr(self, 'close_ary') or len(self.close_ary) == 0:
            print("Warning: No data loaded or empty data")
            return
        
        close_prices = self.close_ary.flatten() if self.close_ary.ndim > 1 else self.close_ary
        daily_returns = np.zeros_like(close_prices)
        daily_returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
            
   
        # Basic volatility
        self.daily_volatility = np.std(daily_returns)
        self.annualized_volatility = self.daily_volatility * np.sqrt(252)
        # Autocorrelation
        returns_series = pd.Series(daily_returns)
        self.autocorr_1d = returns_series.autocorr(lag=1)
        self.autocorr_5d = returns_series.autocorr(lag=5)
        
        # Drawdown
        rolling_max = np.maximum.accumulate(close_prices)
        drawdown = (close_prices / rolling_max - 1.0)
        self.max_drawdown = np.min(drawdown)
        
        # Trend changes
        window_size_short = 20
        window_size_long = 50
        
        ma_short = np.zeros_like(close_prices)
        ma_long = np.zeros_like(close_prices)
        
        for i in range(len(close_prices)):
            if i < window_size_short:
                ma_short[i] = np.mean(close_prices[:i+1])
            else:
                ma_short[i] = np.mean(close_prices[i-window_size_short+1:i+1])
                
            if i < window_size_long:
                ma_long[i] = np.mean(close_prices[:i+1])
            else:
                ma_long[i] = np.mean(close_prices[i-window_size_long+1:i+1])
        
        ma_cross = (ma_short[:-1] > ma_long[:-1]) != (ma_short[1:] > ma_long[1:])
        self.trend_change_frequency = np.sum(ma_cross) / len(ma_cross)


    def load_vnindex_data(self):
        """Load VN-Index data for beta calculation"""
        # Tìm VNINDEX.csv trong parent directories
        current_dir = os.path.dirname(self.data_path)
        vnindex_path = os.path.join(current_dir, "VNINDEX.csv")
        print(f"DEBUG: Looking for VNINDEX at: {vnindex_path}")
        print(f"DEBUG: File exists: {os.path.exists(vnindex_path)}")    
        if os.path.exists(vnindex_path):
            try:
                df = pd.read_csv(vnindex_path)
                
                # Standardize column names
                column_mapping = {
                    'date': 'Date',
                    'close_price': 'Close',
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df = df.rename(columns={old_col: new_col})
                
                # Convert date and sort
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Calculate returns
                df['Returns'] = df['Close'].pct_change()
                df = df.dropna()
                
                self.vnindex_data = df
                print(" VN-Index data loaded successfully for classification")
                return True
                
            except Exception as e:
                print(f"Warning: Could not load VN-Index data: {e}")
                print("Will use alternative cyclical indicators")
                self.vnindex_data = None
                return False
        else:
            print("Warning: VNINDEX.csv not found")
            print("Will use alternative cyclical indicators")
            self.vnindex_data = None
            return False

    def calculate_beta(self, stock_returns, market_returns):
        """Calculate beta coefficient with market"""
        try:
            # Align data by date if both have dates
            stock_returns = stock_returns.dropna()
            market_returns = market_returns.dropna()
            
            if len(stock_returns) < 10 or len(market_returns) < 10:
                return np.nan
            
            # Calculate covariance and variance
            cov_matrix = np.cov(stock_returns, market_returns)
            cov = cov_matrix[0][1]
            var_market = np.var(market_returns)
            
            beta = cov / var_market if var_market > 0 else np.nan
            return beta
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return np.nan

    def classify_stock(self):
        """
        SCIENTIFIC STOCK CLASSIFICATION FOR JOURNAL QUALITY
        Multidimensional approach using statistical validation
        """
        
        print(f"=== SCIENTIFIC CLASSIFICATION: {self.stock_code} ===")
        self.load_vnindex_data()
        # ===== 1. DATA LOADING & VALIDATION =====
        try:
            print("Step 1: Loading full dataset for classification...")
            df_full = pd.read_csv(self.data_path)
            
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
                if old_col in df_full.columns:
                    df_full = df_full.rename(columns={old_col: new_col})
            
            # Date processing
            df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
            df_full = df_full.dropna(subset=['Date'])
            # Handle both string and Timestamp comparison (pandas 2.0+ compatibility)
            try:
                df_full = df_full[(df_full['Date'] >= '2018-01-01') & (df_full['Date'] < '2025-01-01')]
            except TypeError:
                df_full = df_full[(df_full['Date'] >= pd.Timestamp('2018-01-01')) & (df_full['Date'] < pd.Timestamp('2025-01-01'))]
            df_full = df_full.sort_values('Date').reset_index(drop=True)
            
            # Numeric conversion
            for col in ['Close', 'Open', 'High', 'Low']:
                if col in df_full.columns:
                    df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
            
            # Volume handling
            if 'Volume' not in df_full.columns and 'volume_detail' in df_full.columns:
                df_full['Volume'] = pd.to_numeric(df_full['volume_detail'], errors='coerce')
            elif 'Volume' in df_full.columns:
                df_full['Volume'] = pd.to_numeric(df_full['Volume'], errors='coerce')
            else:
                df_full['Volume'] = 1000000
            
            # Calculate returns
            if hasattr(self, 'train_test_split') and self.train_test_split is not None:
                split_date = pd.Timestamp('2024-01-01')
                df_classification = df_full[df_full['Date'] < split_date]  # TRAIN ONLY
                print(f"Classification using TRAIN data: {len(df_classification)} points")
            else:
                # No split specified - use full data (for non-split scenarios)
                df_classification = df_full
                print(f"Classification using FULL data: {len(df_classification)} points")
            
            # Continue with existing classification logic using df_classification
            df_classification['Returns'] = df_classification['Close'].pct_change()
            df = df_classification.dropna()
            
        except Exception as e:
            print(f"ERROR loading data: {e}")
            self.stock_type = "Data_Error"
            return
        
        # ===== 2. DATA QUALITY VALIDATION =====
        validation_results = self._validate_data_quality(df)
        if not validation_results['is_valid']:
            print(f"Data quality insufficient for classification")
            self.stock_type = "Insufficient_Quality"
            return
        
        # ===== 3. COMPUTE CORE METRICS =====
        metrics = self._compute_classification_metrics(df)
        
        # ===== 4. MULTIDIMENSIONAL CLASSIFICATION =====
        classification_result = self._perform_multidimensional_classification(metrics)
        
        # ===== 5. CONFIDENCE ASSESSMENT =====
        confidence_score, confidence_details = self._calculate_classification_confidence(
            validation_results, metrics
        )
        
        # ===== 6. STORE RESULTS =====
        self.stock_type = classification_result['primary_class']
        self.classification_metrics = {
            **metrics,
            'primary_classification': classification_result['primary_class'],
            'secondary_traits': classification_result['secondary_traits'],
            'confidence_score': confidence_score,
            'confidence_details': confidence_details,
            'validation_results': validation_results,
            'data_source': 'FULL_DATASET_SCIENTIFIC'
        }
        
        # Store compatibility metrics
        self.annualized_volatility = metrics['annualized_volatility']
        self.autocorr_1d = metrics['autocorr_1d']
        self.autocorr_5d = metrics['autocorr_5d']
        self.max_drawdown = metrics['max_drawdown']
        self.trend_change_frequency = metrics['trend_change_frequency']
        
        # ===== 7. SCIENTIFIC OUTPUT =====
        self._print_classification_report(classification_result, confidence_score, confidence_details)

    def _validate_data_quality(self, df):
        """Validate data quality for reliable classification"""
        
        results = {
            'is_valid': True,
            'warnings': [],
            'data_length': len(df),
            'date_range_days': (df['Date'].max() - df['Date'].min()).days,
            'missing_data_ratio': df['Returns'].isna().sum() / len(df),
            'outlier_ratio': 0,
            'volatility_stability': 0
        }
        
        # 1. Minimum data requirement
        if len(df) < 252:  # Less than 1 year
            results['is_valid'] = False
            results['warnings'].append(f"Insufficient data: {len(df)} days (minimum 252 required)")
            return results
        
        # 2. Missing data check
        if results['missing_data_ratio'] > 0.05:  # >5% missing
            results['warnings'].append(f"High missing data ratio: {results['missing_data_ratio']:.3f}")
        
        # 3. Outlier detection
        returns = df['Returns'].dropna()
        q99, q1 = returns.quantile(0.99), returns.quantile(0.01)
        outliers = ((returns > q99) | (returns < q1)).sum()
        results['outlier_ratio'] = outliers / len(returns)
        
        if results['outlier_ratio'] > 0.05:
            results['warnings'].append(f"High outlier ratio: {results['outlier_ratio']:.3f}")
        
        # 4. Volatility stability test
        if len(df) >= 504:  # At least 2 years
            mid_point = len(returns) // 2
            vol1 = returns[:mid_point].std() * np.sqrt(252)
            vol2 = returns[mid_point:].std() * np.sqrt(252)
            results['volatility_stability'] = abs(vol1 - vol2) / ((vol1 + vol2) / 2)
            
            if results['volatility_stability'] > 0.5:
                results['warnings'].append(f"Unstable volatility: {results['volatility_stability']:.3f}")
        
        return results

    def _compute_classification_metrics(self, df):
        """Compute all metrics needed for scientific classification"""
        
        returns = df['Returns'].dropna()
        close_prices = df['Close']
        
        # ===== VOLATILITY METRICS =====
        daily_vol = returns.std()
        annualized_volatility = daily_vol * np.sqrt(252)
        
        # Downside volatility
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # ===== RISK METRICS =====
        rolling_max = close_prices.expanding().max()
        drawdown = (close_prices / rolling_max - 1.0)
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_5 = returns.quantile(0.05)
        cvar_5 = returns[returns <= var_5].mean() if (returns <= var_5).any() else var_5
        
        # ===== MOMENTUM/MEAN REVERSION METRICS =====
        autocorr_1d = returns.autocorr(lag=1) if len(returns) > 1 else 0
        autocorr_5d = returns.autocorr(lag=5) if len(returns) > 5 else 0
        autocorr_20d = returns.autocorr(lag=20) if len(returns) > 20 else 0
        
        # Hurst exponent (simplified)
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
        
        # ===== MARKET SENSITIVITY =====
        beta_to_vnindex = np.nan
        if hasattr(self, 'vnindex_data') and self.vnindex_data is not None:
            beta_to_vnindex = self._calculate_beta_with_market(df, self.vnindex_data)
        
        # ===== TREND CHARACTERISTICS =====
        ma_20 = close_prices.rolling(20, min_periods=1).mean()
        ma_50 = close_prices.rolling(50, min_periods=1).mean()
        trend_signal = (ma_20 > ma_50).astype(int)
        trend_changes = trend_signal.diff().abs().sum()
        trend_change_frequency = trend_changes / len(df) if len(df) > 0 else 0
        
        # ===== SKEWNESS AND KURTOSIS =====
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'annualized_volatility': annualized_volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_drawdown,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'autocorr_1d': autocorr_1d,
            'autocorr_5d': autocorr_5d,
            'autocorr_20d': autocorr_20d,
            'hurst_exponent': hurst_exponent,
            'beta_to_vnindex': beta_to_vnindex,
            'trend_change_frequency': trend_change_frequency,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'daily_vol': daily_vol
        }

    def _calculate_beta_with_market(self, stock_df, market_df):
        """Calculate beta with proper error handling"""
        try:
            merged = pd.merge(
                stock_df[['Date', 'Returns']].copy(),
                market_df[['Date', 'Returns']].copy(),
                on='Date', suffixes=('_stock', '_market')
            )
            
            if len(merged) < 30:  # Minimum observations
                return np.nan
            
            stock_returns = merged['Returns_stock'].dropna()
            market_returns = merged['Returns_market'].dropna()
            
            if len(stock_returns) < 30 or len(market_returns) < 30:
                return np.nan
            
            cov_matrix = np.cov(stock_returns, market_returns)
            beta = cov_matrix[0][1] / cov_matrix[1][1] if cov_matrix[1][1] > 0 else np.nan
            return beta
            
        except Exception:
            return np.nan

    def _perform_multidimensional_classification(self, metrics):
        """Multidimensional classification using VN market thresholds"""
        
        # Load VN thresholds (hard-code từ JSON hoặc load file)
        VN_THRESHOLDS = {
            'vol_low': 0.3832, 'vol_high': 0.4768,
            'risk_low': 0.6493, 'risk_high': 0.7754,
            'momentum_low': 0.0321, 'momentum_high': 0.1144,
            'hurst_low': -0.0103, 'hurst_high': 0.0112
        }
        
        # Normalize all metrics (0-1 scale)
        vol_score = self._normalize_score(metrics['annualized_volatility'], 
                                        VN_THRESHOLDS['vol_low'], VN_THRESHOLDS['vol_high'])
        risk_score = self._normalize_score(abs(metrics['max_drawdown']), 
                                        VN_THRESHOLDS['risk_low'], VN_THRESHOLDS['risk_high'])
        momentum_score = self._normalize_score(abs(metrics['autocorr_1d']), 
                                            VN_THRESHOLDS['momentum_low'], VN_THRESHOLDS['momentum_high'])
        hurst_score = self._normalize_score(metrics['hurst_exponent'], 
                                        VN_THRESHOLDS['hurst_low'], VN_THRESHOLDS['hurst_high'])
        
        # WEIGHTED COMPOSITE SCORE
        composite_score = (
            vol_score * 0.4 +      # Volatility chính
            risk_score * 0.3 +     # Risk quan trọng
            momentum_score * 0.2 + # Momentum tendency  
            hurst_score * 0.1      # Mean reversion
        )
        
        # MULTIDIMENSIONAL CLASSIFICATION
        if composite_score > 0.67:
            primary_class = "High_Risk"
        elif composite_score < 0.33:
            primary_class = "Low_Risk"
        else:
            primary_class = "Medium_Risk"
        
        return {
            'primary_class': primary_class,
            'secondary_traits': [],
            'composite_score': composite_score,
            'vol_score': vol_score,
            'risk_score': risk_score,
            'momentum_score': momentum_score,
            'hurst_score': hurst_score
        }

    def _normalize_score(self, value, min_val, max_val):
        """Normalize value to 0-1 scale with bounds checking"""
        if np.isnan(value):
            return 0.5  # neutral score for missing data
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)

    def _calculate_classification_confidence(self, validation_results, metrics):
        """Calculate overall confidence in classification"""
        
        confidence_factors = {}
        
        # Data quality factors
        confidence_factors['data_length'] = min(validation_results['data_length'] / 504, 1.0)  # 2 years = 1.0
        confidence_factors['data_completeness'] = 1.0 - validation_results['missing_data_ratio']
        confidence_factors['outlier_control'] = 1.0 - min(validation_results['outlier_ratio'] * 5, 1.0)
        
        # Stability factors
        if validation_results['volatility_stability'] > 0:
            confidence_factors['volatility_stability'] = 1.0 - min(validation_results['volatility_stability'], 1.0)
        else:
            confidence_factors['volatility_stability'] = 0.8  # Unknown but assume moderate
        
        # Metric reliability factors
        confidence_factors['beta_availability'] = 0.0 if np.isnan(metrics['beta_to_vnindex']) else 1.0
        confidence_factors['autocorr_reliability'] = 0.8 if abs(metrics['autocorr_1d']) < 0.05 else 1.0
        
        # Calculate weighted confidence
        weights = {
            'data_length': 0.25,
            'data_completeness': 0.20,
            'outlier_control': 0.15,
            'volatility_stability': 0.20,
            'beta_availability': 0.10,
            'autocorr_reliability': 0.10
        }
        
        overall_confidence = sum(
            confidence_factors[factor] * weights[factor] 
            for factor in confidence_factors
        )
        
        return overall_confidence, confidence_factors

    def _print_classification_report(self, classification_result, confidence_score, confidence_details):
        """Print comprehensive scientific classification report"""
        
        print(f"\n{'='*60}")
        print(f"SCIENTIFIC STOCK CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(f"Stock Symbol: {self.stock_code}")
        print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nCLASSIFICATION RESULTS:")
        print(f"  Primary Class: {classification_result['primary_class']}")
        if classification_result['secondary_traits']:
            print(f"  Secondary Traits: {', '.join(classification_result['secondary_traits'])}")
        print(f"  Confidence Score: {confidence_score:.3f}/1.000")
        
        print(f"\nKEY METRICS:")
        metrics = self.classification_metrics
        print(f"  Annualized Volatility: {metrics['annualized_volatility']:.4f}")
        print(f"  Maximum Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  Autocorrelation (1d): {metrics['autocorr_1d']:.4f}")
        print(f"  Hurst Exponent: {metrics['hurst_exponent']:.4f}")
        if not np.isnan(metrics['beta_to_vnindex']):
            print(f"  Market Beta: {metrics['beta_to_vnindex']:.4f}")
        else:
            print(f"  Market Beta: N/A (VN-Index data unavailable)")
        
        print(f"\nDATA QUALITY:")
        validation = metrics['validation_results']
        print(f"  Sample Size: {validation['data_length']} trading days")
        print(f"  Missing Data: {validation['missing_data_ratio']:.3f}")
        print(f"  Outlier Ratio: {validation['outlier_ratio']:.3f}")

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm các chỉ báo kỹ thuật vào DataFrame
        """
        # Lưu số lượng dòng ban đầu
        original_rows = len(df)
        
        # Đảm bảo các cột cần thiết tồn tại
        assert 'close_price' in df.columns
        
        # SMA - Simple Moving Average
        df['ma5'] = df['close_price'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close_price'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close_price'].rolling(window=20, min_periods=1).mean()
        
        # Bollinger Bands
        df['ma20'] = df['close_price'].rolling(window=20, min_periods=1).mean()
        df['bb_std'] = df['close_price'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['ma20'] + df['bb_std'] * 2
        df['bb_lower'] = df['ma20'] - df['bb_std'] * 2
        
        # RSI - Relative Strength Index
        delta = df['close_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        df['ema12'] = df['close_price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close_price'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Sử dụng min_periods=1 thay vì loại bỏ NaN để tránh mất dữ liệu
        # Loại bỏ các hàng có giá trị NaN (chỉ các dòng có NaN trong cả các chỉ báo quan trọng)
        important_indicators = ['ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
        df = df.dropna(subset=important_indicators).reset_index(drop=True)
        
        # In thông tin về số dòng bị loại bỏ
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/original_rows*100:.2f}% of data)")
        
        return df
    

    def reset(self):
        """
        Updated reset function to properly initialize all tracking variables
        """
        if self.if_random_reset:
            self.day = np.random.randint(0, min(50, self.max_step // 10))
        else:
            self.day = 0        
        # Initialize portfolio
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
            self.shares = np.array([rd.randint(0, 10)])
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(1, dtype=np.float32)
        
        self.rewards = []
        self.total_fee = 0
        self.days_without_trade = 0
        self.asset_history = [self.initial_amount] 
        self.reward_debug_log = []
        # Calculate initial portfolio value
        self.initial_portfolio_value = (self.close_ary[self.day] * self.shares).sum() + self.amount
        self.total_asset = self.initial_portfolio_value
        
        # Calculate buy and hold return
        self.buy_hold_return = 0.0
        if self.day + self.max_step < len(self.close_ary):
            first_price = self.close_ary[self.day, 0]
            last_price = self.close_ary[self.day + self.max_step - 1, 0]
            self.buy_hold_return = (last_price / first_price - 1)
        
        # Initialize debugging
        if hasattr(self, 'reward_debug_log'):
            self.reward_debug_log = []
        
        # ENHANCED: Initialize all episode tracking variables
        self.reset_episode_tracking()
        
        return self.get_state(), {}
    
    def get_state(self) -> ARY:
        """
        Cung cấp dữ liệu đầu vào cho mạng nơ-ron trong mô hình RL.
        Chuẩn hóa dữ liệu để giúp agent học hiệu quả hơn.
        Kết hợp số tiền, số lượng cổ phiếu, giá cổ phiếu, các chỉ báo kỹ thuật thành một vector trạng thái.        """
        state = np.hstack((
            np.tanh(np.array([self.amount * 2 ** -16])),  # Chuẩn hóa số tiền
            self.shares * 2 ** -9,  # Chuẩn hóa số lượng cổ phiếu
            self.close_ary[self.day] * 2 ** -7,  # Chuẩn hóa giá đóng cửa
            self.tech_ary[self.day] * 2 ** -6,  # Chuẩn hóa chỉ báo kỹ thuật
        ))
        return state

    
    def step(self, action) -> Tuple[ARY, float, bool, bool, Dict]:
        """
        Thực hiện một bước trong môi trường
        """
        # Lưu trạng thái trước khi thực hiện hành động
        prev_total_asset = self.total_asset
        prev_price = self.close_ary[self.day, 0]
        prev_shares = self.shares[0]  # Lưu số lượng cổ phiếu trước đó
        
        # Khởi tạo debug_info
        debug_info = {}
        debug_info["day"] = self.day
        debug_info["date"] = str(self.dates[self.day])[:10]
        debug_info["price"] = prev_price
        debug_info["prev_price"] = prev_price
        debug_info["prev_shares"] = self.shares[0]
        debug_info["prev_amount"] = self.amount
        debug_info["prev_total_asset"] = prev_total_asset
        
        self.day += 1
        # Kiểm tra nếu day vượt quá giới hạn của dữ liệu
        if self.day >= len(self.close_ary):
            self.day = len(self.close_ary) - 1
            terminal = True  # Kết thúc episode
        else:
            terminal = self.day == self.max_step
        
        # Xử lý hành động
        action = action.copy()
        action = action[0]  # Chỉ lấy hành động đầu tiên vì chỉ có 1 cổ phiếu
        
        debug_info["raw_action"] = action
        # Kiểm tra NaN và xử lý
        if np.isnan(action):
            action = 0.0  # Hoặc một giá trị mặc định khác phù hợp
            print("Warning: Detected NaN in action, setting to 0")
        # Giảm ngưỡng bỏ qua hành động để khuyến khích giao dịch
        if -0.1 < action < 0.1:  # Reduced from 0.25
            action = 0

        
        # Chuyển đổi hành động thành số lượng cổ phiếu cần giao dịch
        stock_action = self.improved_action_scaling(action)
        debug_info["stock_action"] = stock_action
        
        # Lấy giá cổ phiếu hiện tại
        adj_close_price = self.close_ary[self.day, 0]
        price_change = adj_close_price - prev_price
        price_change_pct = price_change / prev_price if prev_price > 0 else 0
        debug_info["curr_price"] = adj_close_price
        debug_info["price_change_pct"] = price_change_pct
        
        # Biến để theo dõi phí giao dịch trong bước hiện tại
        current_fee = 0
        
        # Thực hiện hành động giao dịch
        if stock_action > 0:  # Mua cổ phiếu
            delta_stock = min(self.amount // adj_close_price, stock_action)
            current_fee = adj_close_price * delta_stock * self.cost_pct  # Tính phí
            self.amount -= adj_close_price * delta_stock * (1 + self.cost_pct)
            self.shares[0] += delta_stock
            debug_info["trade_type"] = "BUY"
            debug_info["delta_stock"] = delta_stock
            debug_info["current_fee"] = current_fee  # Lưu phí hiện tại
            
            # Cập nhật phí tích lũy
            if not hasattr(self, 'cumulative_fee'):
                self.cumulative_fee = 0
            self.cumulative_fee += current_fee
            
        elif stock_action < 0 and self.shares[0] > 0:  # Bán cổ phiếu
            delta_stock = min(-stock_action, self.shares[0])
            current_fee = adj_close_price * delta_stock * self.cost_pct  # Tính phí
            self.amount += adj_close_price * delta_stock * (1 - self.cost_pct)
            self.shares[0] -= delta_stock
            debug_info["trade_type"] = "SELL"
            debug_info["delta_stock"] = delta_stock
            debug_info["current_fee"] = current_fee  # Lưu phí hiện tại
            
            # Cập nhật phí tích lũy
            if not hasattr(self, 'cumulative_fee'):
                self.cumulative_fee = 0
            self.cumulative_fee += current_fee
            
        else:
            debug_info["trade_type"] = "HOLD"
            debug_info["current_fee"] = 0
        
        # Tính toán tổng tài sản mới
        new_total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        debug_info["new_shares"] = self.shares[0]
        debug_info["new_amount"] = self.amount
        debug_info["new_total_asset"] = new_total_asset

        # Cập nhật tổng phí giao dịch
        if not hasattr(self, 'total_fee'):
            self.total_fee = 0
        self.total_fee += current_fee
        
        # === TÍNH REWARD BẰNG PHƯƠNG THỨC ĐÃ TÁCH ===
        reward, reward_components = self.calculate_reward(
            action=action,
            price_change_pct=price_change_pct,
            current_total_asset=new_total_asset,
            prev_total_asset=prev_total_asset,
            current_position=self.shares[0],
            fee_paid=current_fee
        )
        
        # Thêm thông tin về các thành phần reward vào debug_info
        for component_name, component_value in reward_components.items():
            debug_info[f"reward_{component_name}"] = component_value
        
        # Lưu phần thưởng
        if not hasattr(self, 'rewards') or self.rewards is None:
            self.rewards = []
        self.rewards.append(reward)

        # Thêm phần thưởng bổ sung ở cuối nếu kết thúc
        if terminal:
            # Tính lợi nhuận cuối cùng
            self.cumulative_returns = (new_total_asset / self.initial_portfolio_value - 1.0) * 100
            debug_info["final_return"] = self.cumulative_returns
            
            # Thêm thông tin về phí giao dịch vào debug_info khi kết thúc episode
            debug_info["total_fee"] = self.total_fee
            debug_info["fee_percentage"] = (self.total_fee / self.initial_portfolio_value) * 100
            debug_info["total_trades"] = len(self.reward_debug_log) if hasattr(self, 'reward_debug_log') else 0

        # Lưu debug info
        if hasattr(self, 'reward_debug_log'):
            self.reward_debug_log.append(debug_info)
        else:
            self.reward_debug_log = [debug_info]

        # Cập nhật tổng tài sản
        self.total_asset = new_total_asset

        # Lấy trạng thái mới
        state = self.get_state()
        truncated = False
                
        return state, reward, terminal, truncated, {}
    def calculate_reward(self, action, price_change_pct, current_total_asset, 
                            prev_total_asset, current_position, fee_paid):
            """
            SCIENTIFICALLY OPTIMIZED REWARD FUNCTION
            - Low_Risk: ổn định, khuyến khích momentum nhẹ
            - Medium_Risk: mean reversion và regime detection
            - High_Risk: volatility-aware adaptive reward (Việt Nam-specific)
            """

            # === BASIC CALCULATIONS ===
            asset_change_pct = (current_total_asset - prev_total_asset) / prev_total_asset if prev_total_asset > 0 else 0
            fee_ratio = fee_paid / prev_total_asset if prev_total_asset > 0 else 0
            abs_price_change = abs(price_change_pct)

            # === DEFAULT VALUES ===
            total_reward = 0.0
            reward_components = {}

            # =========================================================
            #  LOW-RISK: GIỮ NGUYÊN LOGIC GỐC
            # =========================================================

            if self.stock_type == 'Low_Risk':
                # ============================================================
                # LOW-RISK REWARD FUNCTION - OPTIMIZED VERSION
                # Fixes: Overtrading problem, unstable learning
                # Data: 83 VN stocks (Vol 18-40%)
                # ============================================================
                
                # DATA-DRIVEN THRESHOLDS
                micro_threshold = 0.001100    # 0.11%
                tiny_threshold = 0.004975     # 0.50%
                small_threshold = 0.010377    # 1.04%
                medium_threshold = 0.016970   # 1.70%
                large_threshold = 0.032173    # 3.22%
                extreme_threshold = 0.059701  # 5.97%
                
                # Initialize history
                if not hasattr(self, 'low_risk_history'):
                    self.low_risk_history = {
                        'sharpe_history': {'mean': 0, 'std': 0.01, 'sharpe': 0},
                        'recent_returns': [],
                        'trend_window': [],
                        'max_asset': current_total_asset,
                        'trade_count': 0,
                        'steps': 0
                    }
                
                # ============================================================
                # COMPONENT 1: PROFIT-BASED (40%)
                # Reduced from 45% - focus on consistency over step profits
                # ============================================================
                
                # Base profit - REDUCED multiplier
                base_profit = asset_change_pct * 70
                
                # Consistency bonus - STRICTER requirements
                self.low_risk_history['recent_returns'].append(asset_change_pct)
                if len(self.low_risk_history['recent_returns']) > 10:
                    self.low_risk_history['recent_returns'].pop(0)
                
                if len(self.low_risk_history['recent_returns']) >= 5:
                    returns_std = np.std(self.low_risk_history['recent_returns'])
                    
                    if returns_std < 0.003:  # Very stable
                        consistency_bonus = 0.25
                    elif returns_std < 0.008:  # Moderate
                        consistency_bonus = 0.12
                    else:  # Too volatile
                        consistency_bonus = -0.10
                else:
                    consistency_bonus = 0
                
                # Differential Sharpe - INCREASED weight
                DECAY = 0.97
                sh = self.low_risk_history['sharpe_history']
                
                new_mean = DECAY * sh['mean'] + (1-DECAY) * asset_change_pct
                new_std_sq = DECAY * sh['std']**2 + (1-DECAY) * (asset_change_pct - new_mean)**2
                new_std = np.sqrt(max(new_std_sq, 1e-8))
                new_sharpe = new_mean / (new_std + 1e-6)
                
                sharpe_delta = (new_sharpe - sh['sharpe']) * 55  # Increased from 45
                
                self.low_risk_history['sharpe_history'] = {
                    'mean': new_mean,
                    'std': new_std,
                    'sharpe': new_sharpe
                }
                
                profit_reward = base_profit + consistency_bonus + sharpe_delta
                
                # ============================================================
                # COMPONENT 2: TREND-FOLLOWING (25%)
                # Reduced from 30% - less incentive for micro-moves
                # ============================================================
                
                # Track trend
                self.low_risk_history['trend_window'].append(price_change_pct)
                if len(self.low_risk_history['trend_window']) > 20:
                    self.low_risk_history['trend_window'].pop(0)
                
                if len(self.low_risk_history['trend_window']) >= 10:
                    recent_trend = np.mean(self.low_risk_history['trend_window'])
                    if recent_trend > 0.002:
                        trend_dir = 'up'
                    elif recent_trend < -0.002:
                        trend_dir = 'down'
                    else:
                        trend_dir = 'neutral'
                else:
                    trend_dir = 'neutral'
                
                # Reward structure - REDUCED for small moves
                if abs_price_change > extreme_threshold:
                    # EXTREME (5.97%+)
                    if trend_dir == 'up' and action > 0.12:
                        trend_reward = 55
                    elif trend_dir == 'down' and action < -0.12:
                        trend_reward = 50
                    else:
                        trend_reward = 25
                
                elif abs_price_change > large_threshold:
                    # LARGE (3.22-5.97%)
                    if trend_dir == 'up' and action > 0.08:
                        trend_reward = 42
                    elif trend_dir == 'down' and action < -0.08:
                        trend_reward = 38
                    else:
                        trend_reward = 20
                
                elif abs_price_change > medium_threshold:
                    # MEDIUM (1.70-3.22%)
                    if trend_dir == 'up' and action > 0.05:
                        trend_reward = 32
                    elif trend_dir == 'down' and action < -0.05:
                        trend_reward = 28
                    else:
                        trend_reward = 15
                
                elif abs_price_change > small_threshold:
                    # SMALL (1.04-1.70%)
                    if trend_dir != 'neutral' and abs(action) > 0.03:
                        trend_reward = 18
                    else:
                        trend_reward = 10
                
                elif abs_price_change > tiny_threshold:
                    # TINY (0.50-1.04%) - REDUCED rewards
                    if abs(action) < 0.02:
                        trend_reward = 12  # Was 16
                    elif abs(action) < 0.04:
                        trend_reward = 7   # Was 10
                    else:
                        trend_reward = 0   # Was 3
                
                elif abs_price_change > micro_threshold:
                    # MICRO (0.11-0.50%) - STRONG patience required
                    if abs(action) < 0.01:
                        trend_reward = 8   # Was 14
                    elif abs(action) < 0.03:
                        trend_reward = 2   # Was 7
                    else:
                        trend_reward = -5  # ADDED penalty
                
                else:
                    # ULTRA-MICRO (<0.11%) - HEAVY penalty for trading
                    if abs(action) < 0.01:
                        trend_reward = 5   # Was 10
                    else:
                        trend_reward = -8  # Was -4
                
                # ============================================================
                # COMPONENT 3: RISK CONTROL (15%)
                # Unchanged weight but stricter thresholds
                # ============================================================
                
                self.low_risk_history['max_asset'] = max(
                    self.low_risk_history['max_asset'], 
                    current_total_asset
                )
                
                if self.low_risk_history['max_asset'] > 0:
                    current_dd = (current_total_asset / self.low_risk_history['max_asset'] - 1.0)
                else:
                    current_dd = 0
                
                # Progressive DD penalty
                if current_dd < -0.12:
                    risk_penalty = -0.30
                elif current_dd < -0.10:
                    risk_penalty = -0.20
                elif current_dd < -0.07:
                    risk_penalty = -0.12
                elif current_dd < -0.05:
                    risk_penalty = -0.06
                elif current_dd < -0.03:
                    risk_penalty = -0.02
                else:
                    risk_penalty = 0.06
                
                # Position size - STRICTER
                position_value = current_position * self.close_ary[self.day, 0] if self.day < len(self.close_ary) else 0
                position_ratio = position_value / current_total_asset if current_total_asset > 0 else 0
                
                if position_ratio > 0.80:  # Was 0.85
                    position_penalty = -0.15  # Was -0.12
                elif position_ratio > 0.60:  # Was 0.65
                    position_penalty = -0.06  # Was -0.04
                elif 0.40 < position_ratio < 0.60:
                    position_penalty = 0.08
                else:
                    position_penalty = -0.04
                
                risk_reward = risk_penalty + position_penalty
                
                # ============================================================
                # COMPONENT 4: TRANSACTION EFFICIENCY (20%)
                # DOUBLED from 10% - CRITICAL FIX for overtrading
                # ============================================================
                
                self.low_risk_history['steps'] += 1
                
                # MUCH STRICTER thresholds
                extreme_fee = 0.005
                very_heavy = 0.003
                heavy = 0.0015
                moderate = 0.0008
                
                if fee_ratio > extreme_fee:
                    efficiency_reward = -0.50  # Was -0.28
                    self.low_risk_history['trade_count'] += 1
                
                elif fee_ratio > very_heavy:
                    if abs_price_change > large_threshold:
                        efficiency_reward = -0.15
                    else:
                        efficiency_reward = -0.40
                    self.low_risk_history['trade_count'] += 1
                
                elif fee_ratio > heavy:
                    if abs_price_change > medium_threshold:
                        efficiency_reward = 0.00  # Neutral
                    else:
                        efficiency_reward = -0.30
                    self.low_risk_history['trade_count'] += 1
                
                elif fee_ratio > moderate:
                    if abs_price_change > small_threshold:
                        efficiency_reward = 0.05
                    else:
                        efficiency_reward = -0.20
                    self.low_risk_history['trade_count'] += 1
                
                else:
                    # No trading - STRONG BONUS
                    if abs_price_change < tiny_threshold:
                        efficiency_reward = 0.20  # Was 0.12 - DOUBLED
                    elif abs_price_change > large_threshold:
                        efficiency_reward = -0.12
                    else:
                        efficiency_reward = 0.12
                
                # Frequency penalty - CHECK EARLIER
                if self.low_risk_history['steps'] > 40:  # Was 60
                    monthly_trades = self.low_risk_history['trade_count'] / (self.low_risk_history['steps'] / 20)
                    
                    if monthly_trades > 8:
                        efficiency_reward -= 0.20  # Was -0.10
                    elif monthly_trades > 5:
                        efficiency_reward -= 0.10  # Was -0.04
                    elif 2 < monthly_trades < 5:
                        efficiency_reward += 0.08  # Was +0.04
                    elif monthly_trades < 2:
                        efficiency_reward -= 0.05  # New penalty
                
                # ============================================================
                # COMBINE - NEW WEIGHTS
                # ============================================================
                
                total_reward = (
                    profit_reward * 0.40 +      # Was 0.45
                    trend_reward * 0.25 +        # Was 0.30
                    risk_reward * 0.15 +         # Unchanged
                    efficiency_reward * 0.20     # Was 0.10 - DOUBLED
                )
                
                # STRICTER clipping
                total_reward = np.clip(total_reward, -3.0, 3.0)
                
                reward_components = {
                    "strategy": "Low_Risk_AntiOvertrading_v2",
                    "profit_reward": profit_reward,
                    "trend_reward": trend_reward,
                    "risk_reward": risk_reward,
                    "efficiency_reward": efficiency_reward,
                    "data_source": "83_VN_low_risk_stocks_fixed",
                    "version": "v2_overtrading_fixed",
                    "micro_threshold": micro_threshold,
                    "tiny_threshold": tiny_threshold,
                    "small_threshold": small_threshold,
                    "medium_threshold": medium_threshold,
                    "large_threshold": large_threshold
                }


            elif self.stock_type == 'Medium_Risk':
                # ============================================================
                # MEDIUM-RISK V3 - ANTI-OVERTRADING FOCUS
                
                # ============================================================
                
                SMALL_MOVE = 0.008811
                MEDIUM_MOVE = 0.016854
                LARGE_MOVE = 0.029412
                
                # ===== INITIALIZE =====
                if not hasattr(self, 'medium_risk_state'):
                    self.medium_risk_state = {
                        'ema_return': 0.0,
                        'ema_variance': 0.001,
                        'prev_sharpe': 0.0,
                        'return_history': [],
                        'price_history': [],
                        'peak_portfolio': current_total_asset,
                        'total_trades': 0,
                        'steps': 0,
                        'days_since_last_trade': 0  # NEW: Track patience
                    }
                
                state = self.medium_risk_state
                state['steps'] += 1
                
                # ============================================================
                # COMPONENT 1: PROFIT REWARD (30% - GIẢM TỪ 35%)
                # ============================================================
                
                # Base profit (unchanged)
                if asset_change_pct > 0:
                    base_profit = np.sqrt(abs(asset_change_pct)) * np.sign(asset_change_pct) * 70  # GIẢM từ 80
                else:
                    base_profit = asset_change_pct * 80  # GIẢM từ 90
                
                # Differential Sharpe
                α = 0.05
                state['ema_return'] = (1 - α) * state['ema_return'] + α * asset_change_pct
                state['ema_variance'] = (1 - α) * state['ema_variance'] + α * (asset_change_pct - state['ema_return'])**2
                
                ema_std = np.sqrt(max(state['ema_variance'], 1e-8))
                new_sharpe = state['ema_return'] / (ema_std + 1e-6)
                sharpe_improvement = (new_sharpe - state['prev_sharpe']) * 35  # GIẢM từ 40
                state['prev_sharpe'] = new_sharpe
                
                profit_reward = base_profit + sharpe_improvement
                
                # ============================================================
                # COMPONENT 2: STRATEGY REWARD (25% - GIẢM TỪ 35%)
                # CRITICAL: Chỉ reward HIGH-CONVICTION moves
                # ============================================================
                
                state['price_history'].append(price_change_pct)
                if len(state['price_history']) > 25:
                    state['price_history'].pop(0)
                
                # Calculate conviction score
                conviction_score = 0.0
                
                if abs_price_change > LARGE_MOVE:
                    # LARGE MOVE: High conviction only
                    if price_change_pct > 0 and action < -0.12:  # TĂNG threshold từ -0.1
                        conviction_score = 1.0
                    elif price_change_pct < 0 and action > 0.12:
                        conviction_score = 0.95
                    elif abs(action) < 0.05:  # REWARD patience even on large moves
                        conviction_score = 0.4
                    else:
                        conviction_score = 0.2  # GIẢM từ 0.5
                
                elif abs_price_change > MEDIUM_MOVE:
                    # MEDIUM MOVE: Moderate conviction
                    if abs(action) > 0.12:  # TĂNG threshold
                        if price_change_pct * action < 0:  # Contrarian
                            conviction_score = 0.65  # GIẢM từ 0.75
                        else:
                            conviction_score = 0.3
                    elif abs(action) < 0.05:
                        conviction_score = 0.45
                    else:
                        conviction_score = 0.25
                
                elif abs_price_change > SMALL_MOVE:
                    # SMALL MOVE: Prefer patience
                    if abs(action) < 0.08:  # Patient
                        conviction_score = 0.6  # GIẢM từ 0.65
                    else:
                        conviction_score = 0.15  # HEAVY penalty
                
                else:
                    # TINY MOVE: STRONG patience requirement
                    if abs(action) < 0.03:
                        conviction_score = 0.55
                    elif abs(action) < 0.08:
                        conviction_score = 0.2
                    else:
                        conviction_score = -0.1  # NEGATIVE for trading on noise
                
                strategy_reward = conviction_score * 35  # GIẢM multiplier từ 45
                
                # ============================================================
                # COMPONENT 3: RISK CONTROL (20% - GIỮ NGUYÊN)
                # ============================================================
                
                # Drawdown
                state['peak_portfolio'] = max(state['peak_portfolio'], current_total_asset)
                current_dd = (current_total_asset / state['peak_portfolio'] - 1.0)
                
                if current_dd < -0.12:
                    dd_penalty = -0.35
                elif current_dd < -0.08:
                    dd_penalty = -0.18
                elif current_dd < -0.04:
                    dd_penalty = -0.06
                else:
                    dd_penalty = 0.04
                
                # Position sizing (unchanged but stricter)
                position_value = current_position * self.close_ary[self.day, 0] if self.day < len(self.close_ary) else 0
                position_ratio = position_value / current_total_asset if current_total_asset > 0 else 0
                
                if 0.45 <= position_ratio <= 0.65:  # TIGHTEN ideal range
                    position_reward = 0.12
                elif 0.30 <= position_ratio < 0.45:
                    position_reward = 0.06
                elif 0.65 < position_ratio <= 0.80:
                    position_reward = 0.02
                else:
                    position_reward = -0.15
                
                risk_reward = dd_penalty + position_reward
                
                # ============================================================
                # COMPONENT 4: TRANSACTION EFFICIENCY (25% - TĂNG TỪ 10%)
                # CRITICAL ANTI-OVERTRADING MECHANISM
                # ============================================================
                
                HEAVY_FEE = 0.003
                MODERATE_FEE = 0.001
                
                # 4.1 Immediate fee penalty
                if fee_ratio > HEAVY_FEE:
                    if abs_price_change > LARGE_MOVE:
                        fee_penalty = -0.05  # Still penalty even on large move
                    else:
                        fee_penalty = -0.40  # TĂNG từ -0.25
                    state['total_trades'] += 1
                    state['days_since_last_trade'] = 0
                
                elif fee_ratio > MODERATE_FEE:
                    if abs_price_change > MEDIUM_MOVE:
                        fee_penalty = 0.00  # Neutral
                    else:
                        fee_penalty = -0.25  # TĂNG từ -0.10
                    state['total_trades'] += 1
                    state['days_since_last_trade'] = 0
                
                else:
                    # NO TRADING - STRONG REWARD
                    state['days_since_last_trade'] += 1
                    
                    if abs_price_change < SMALL_MOVE:
                        fee_penalty = 0.20  # TĂNG từ 0.12 - REWARD patience heavily
                    elif abs_price_change > LARGE_MOVE:
                        fee_penalty = -0.15  # Penalty for missing opportunity
                    else:
                        fee_penalty = 0.10  # TĂNG từ 0.06
                
                # 4.2 Patience bonus (NEW)
                if state['days_since_last_trade'] > 5:
                    patience_bonus = 0.08
                elif state['days_since_last_trade'] > 3:
                    patience_bonus = 0.04
                else:
                    patience_bonus = 0.0
                
                # 4.3 Frequency penalty (STRENGTHENED)
                frequency_penalty = 0.0
                if state['steps'] > 50:
                    trading_freq = state['total_trades'] / state['steps']
                    
                    if trading_freq > 0.30:  # >30% days
                        frequency_penalty = -0.25  # TĂNG từ -0.12
                    elif trading_freq > 0.20:  # 20-30%
                        frequency_penalty = -0.12  # TĂNG từ -0.05
                    elif trading_freq > 0.15:  # 15-20%
                        frequency_penalty = -0.04
                    elif 0.08 < trading_freq < 0.15:  # Sweet spot: 8-15%
                        frequency_penalty = 0.08  # REWARD
                    elif trading_freq < 0.05:  # Too passive
                        frequency_penalty = -0.05
                
                efficiency_reward = fee_penalty + patience_bonus + frequency_penalty
                
                # ============================================================
                # FINAL COMBINATION (NEW WEIGHTS)
                # ============================================================
                
                total_reward = (
                    profit_reward * 0.28 +       # 0.30 → 0.28
                    strategy_reward * 0.30 +     # 0.25 → 0.30
                    risk_reward * 0.18 +         # 0.20 → 0.18
                    efficiency_reward * 0.24     # 0.25 → 0.24
                )
                
                # Adaptive clipping
                if len(state['price_history']) >= 20:
                    realized_vol = np.std(state['price_history']) * np.sqrt(252)
                    if realized_vol > 0.50:
                        clip_bound = 3.0  # GIẢM từ 3.5
                    elif realized_vol < 0.35:
                        clip_bound = 1.8  # GIẢM từ 2.0
                    else:
                        clip_bound = 2.2  # GIẢM từ 2.5
                else:
                    clip_bound = 2.2
                
                total_reward = np.clip(total_reward, -clip_bound, clip_bound)
                
                reward_components = {
                    "strategy": "Medium_Risk_AntiOvertrading_v3",
                    "profit_reward": profit_reward,
                    "strategy_reward": strategy_reward,
                    "risk_reward": risk_reward,
                    "efficiency_reward": efficiency_reward,
                    "fee_penalty": fee_penalty,
                    "patience_bonus": patience_bonus,
                    "frequency_penalty": frequency_penalty,
                    "trading_frequency": state['total_trades'] / state['steps'] if state['steps'] > 0 else 0,
                    "days_since_last_trade": state['days_since_last_trade'],
                    "conviction_score": conviction_score,
                    "version": "v3_anti_overtrading"
                }


            # =========================================================
            #HIGH-RISK: VOLATILITY-AWARE ADAPTIVE REWARD (VIETNAM)
            # =========================================================

            elif self.stock_type == 'High_Risk':
  
                
                extreme_threshold = 0.025
                medium_threshold = 0.012
                
                # ============================================================
                # COMPONENT 1: MOMENTUM (60% - GIỮ NGUYÊN)
                # MODERATE multipliers - không quá aggressive
                # ============================================================
                
                momentum_reward = 0.0
                
                if abs_price_change > extreme_threshold:  # >2.5%
                    # MODERATE increase: 85 → 100 (chỉ +18%, không +41%)
                    momentum_mult_extreme = 95  # Was 85 (v1), 120 (v2)
                    momentum_mult_contrarian = 71  # Was 35 (v1), 50 (v2)
                    
                    if price_change_pct > 0:  # Tăng mạnh
                        if action > 0.3:  # Mua mạnh
                            momentum_reward = asset_change_pct * momentum_mult_extreme
                        elif action > 0.1:
                            momentum_reward = asset_change_pct * (momentum_mult_extreme * 0.8)
                        elif action < -0.2:  # Chốt lời
                            momentum_reward = asset_change_pct * momentum_mult_contrarian
                        else:
                            momentum_reward = asset_change_pct * 28  # Was 25/30
                    else:  # Giảm mạnh
                        if action < -0.3:
                            momentum_reward = -asset_change_pct * (momentum_mult_extreme * 0.9)
                        elif action < -0.1:
                            momentum_reward = -asset_change_pct * (momentum_mult_extreme * 0.7)
                        elif action > 0.15:  # Mua đáy
                            momentum_reward = asset_change_pct * momentum_mult_contrarian
                        else:
                            momentum_reward = asset_change_pct * 32
                
                elif abs_price_change > medium_threshold:  # 1.2-2.5%
                    # MODERATE: 65 → 75 (chỉ +15%, không +31%)
                    momentum_mult_medium = 75  # Was 65 (v1), 85 (v2)
                    
                    if price_change_pct > 0:
                        if action > 0.2:
                            momentum_reward = asset_change_pct * momentum_mult_medium
                        elif action > 0.1:
                            momentum_reward = asset_change_pct * (momentum_mult_medium * 0.8)
                        else:
                            momentum_reward = asset_change_pct * (momentum_mult_medium * 0.6)
                    else:
                        if action < -0.2:
                            momentum_reward = -asset_change_pct * (momentum_mult_medium * 0.9)
                        elif action < -0.1:
                            momentum_reward = -asset_change_pct * (momentum_mult_medium * 0.7)
                        else:
                            momentum_reward = asset_change_pct * (momentum_mult_medium * 0.5)
                
                else:  # Small moves
                    if abs(action) < 0.15:
                        momentum_reward = 0.09  # Was 0.08/0.10
                    elif abs(action) > 0.4:
                        momentum_reward = -0.11  # Was -0.12/-0.10
                    else:
                        momentum_reward = 0.04  # Was 0.03/0.05
                
                # ============================================================
                # COMPONENT 2: VOLATILITY EXPLOITATION (25%)
                # ============================================================
                
                if not hasattr(self, 'recent_returns'):
                    self.recent_returns = []
                
                self.recent_returns.append(price_change_pct)
                if len(self.recent_returns) > 10:
                    self.recent_returns.pop(0)
                
                volatility_reward = 0.0
                
                if len(self.recent_returns) >= 5:
                    recent_vol = np.std(self.recent_returns)
                    
                    if recent_vol > 0.030:  # High vol
                        if 0.2 < abs(action) < 0.5:  # GIẢM từ 0.6 xuống 0.5
                            volatility_reward = 0.13  # Was 0.12/0.15
                        elif abs(action) > 0.5:  # Was 0.6
                            volatility_reward = -0.04  # Was -0.05/-0.03
                        else:
                            volatility_reward = -0.09  # Was -0.08/-0.10
                    elif recent_vol > 0.018:
                        volatility_reward = 0.09 if 0.1 < abs(action) < 0.4 else 0.03
                    else:
                        volatility_reward = 0.06 if abs(action) < 0.3 else -0.02
                
                # ============================================================
                # COMPONENT 3: TRADING MANAGEMENT (15%)
                # MODERATE tolerance
                # ============================================================
                
                # MODERATE tolerance: không quá lỏng
                fee_tolerance_high = 0.0045  # Was 0.004/0.005 - MIDDLE
                fee_tolerance_med = 0.00175  # Was 0.0015/0.002 - MIDDLE
                
                if fee_ratio > fee_tolerance_high:
                    if abs_price_change > extreme_threshold:
                        trading_reward = 0.08  # Was 0.08/0.10
                    else:
                        trading_reward = -0.13  # Was -0.12/-0.10
                elif fee_ratio > fee_tolerance_med:
                    trading_reward = 0.07 if abs_price_change > medium_threshold else -0.05
                else:
                    if abs_price_change < medium_threshold:
                        trading_reward = 0.07
                    else:
                        trading_reward = -0.01  # Was -0.02/0.00
                
                # ============================================================
                # COMBINE
                # ============================================================
                
                total_reward = (
                    momentum_reward * 0.60 +      # GIỮ NGUYÊN
                    volatility_reward * 0.25 +    # GIỮ NGUYÊN
                    trading_reward * 0.15         # GIỮ NGUYÊN
                )
                
                # MODERATE clipping
                total_reward = np.clip(total_reward, -2.8, 2.8)  # Was -2.5/-3.5
                
                reward_components = {
                    "strategy": "High_Risk_Balanced_v3",
                    "momentum_reward": momentum_reward,
                    "volatility_reward": volatility_reward,
                    "trading_reward": trading_reward,
                    "approach": "moderate_aggressive",
                    "version": "v3_balanced"
                }
            # =========================================================
            #  FINAL SCALING & RETURN (CHUNG CHO TẤT CẢ)
            # =========================================================
            scaled_reward = total_reward * self.reward_scale

            reward_components["final_reward"] = scaled_reward

            return scaled_reward, reward_components


    def reset_episode_tracking(self):
        """Reset episode tracking for FPT-specific features"""
        self.max_portfolio_in_episode = self.total_asset
        
        # FPT-specific tracking
        self.recent_prices = []  # For BB calculation
        
        # Initialize other tracking if needed
        if not hasattr(self, 'total_fee'):
            self.total_fee = 0
        if not hasattr(self, 'cumulative_fee'):
            self.cumulative_fee = 0

    def improved_action_scaling(self, raw_action):
        """
        SIMPLIFIED ACTION SCALING - Remove complexity
        """
        
        # === SIMPLE DEADBAND ===
        if abs(raw_action) < 0.1:  # Reduced threshold
            return 0.0
        
        # === BASIC SCALING ===
        portfolio_value = self.total_asset
        current_price = self.close_ary[self.day, 0]
        
        # Simple percentage-based trading
        if self.stock_type == 'Low_Risk':
            max_trade_pct = 0.15  # 15% of portfolio max
        else:
            max_trade_pct = 0.22
        
        # Calculate trade size
        max_trade_value = portfolio_value * max_trade_pct
        max_shares = max_trade_value / current_price
        
        # Simple linear scaling
        scaled_shares = raw_action * max_shares
        
        # Conservative limit
        scaled_shares = np.clip(scaled_shares, -max_shares, max_shares)
        
        return int(scaled_shares)


    def calculate_position_target(self, action):
        """
        Instead of direct trading, calculate target position
        This prevents over-trading seen in your results
        """
        current_position_ratio = (self.shares[0] * self.close_ary[self.day, 0]) / self.total_asset
        
        # Target position based on action
        if action > 0.5:
            target_ratio = 0.8
        elif action > 0.2:
            target_ratio = 0.6
        elif action > -0.2:
            target_ratio = current_position_ratio  # Hold
        elif action > -0.5:
            target_ratio = 0.3
        else:
            target_ratio = 0.1
        
        # Calculate required trade to reach target
        target_value = self.total_asset * target_ratio
        current_value = self.shares[0] * self.close_ary[self.day, 0]
        trade_value = target_value - current_value
        
        # Convert to shares
        shares_to_trade = trade_value / self.close_ary[self.day, 0]
        
        return int(shares_to_trade)
def get_stock_codes_from_directory(directory_path):
    """
    Lấy danh sách mã cổ phiếu từ các tên file trong thư mục
    
    Args:
        directory_path: Đường dẫn đến thư mục chứa các file dữ liệu
        
    Returns:
        List[str]: Danh sách các mã cổ phiếu
    """
    stock_codes = []
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            stock_code = file.split('.')[0]
            stock_codes.append(stock_code)
    return stock_codes
    
def test_stock_trading_env(stock_code=None, data_path=None):
    """
    Kiểm tra môi trường giao dịch cổ phiếu
    
    Args
        stock_code: Mã cổ phiếu để kiểm tra (nếu None thì dùng)
        data_path: Đường dẫn đến file dữ liệu (nếu None thì tự tạo dựa vào stock_code)
    """
    
    if data_path is None:
        data_path = f'../trainingset/{stock_code}.csv'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    
    env = StockTradingEnv(data_path=data_path, stock_code=stock_code)
    env.if_random_reset = False #dont random set
    evaluate_time = 4 #Check all data in 4 iterations

    print(f"\nTest {stock_code} with random action:")
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = np.array([rd.uniform(-1, +1)])
        state, reward, terminal, truncated, info_dict = env.step(action)
        done = terminal or truncated
        if done:
            print(f'{stock_code} - Acumulated profit: {env.cumulative_returns:9.2f}%')
            state, info_dict = env.reset()

    print(f"\nTest {stock_code} with buy and hold strategy:")
    env.if_random_reset = True
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = np.array([1.0])  # Luôn mua cổ phiếu
        state, reward, terminal, truncated, info_dict = env.step(action)
        done = terminal or truncated
        if done:
            print(f'{stock_code} - Acumulated profit: {env.cumulative_returns:9.2f}%')
            state, info_dict = env.reset()

def test_multiple_stocks(directory_path = '../trainingset/'):
    """
    Kiểm tra môi trường giao dịch cho nhiều mã cổ phiếu
    
    Args:
        directory_path: Đường dẫn đến thư mục chứa dữ liệu cổ phiếu
    """

    stock_codes = get_stock_codes_from_directory(directory_path)

    if not stock_codes:
        print(f"Not find file data in folder: {directory_path}")
        return
    print(f"Find {len(stock_codes)} stock: {','.join(stock_codes)}")

    for stock_code in stock_codes:
        data_path = f'{directory_path}/{stock_code}.csv'
        test_stock_trading_env(stock_code, data_path)
        print("-" * 60)