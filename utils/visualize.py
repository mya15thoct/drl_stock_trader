import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def analyze_trading_results(
    prices: List[float],
    actions: List[float],
    portfolio_values: List[float],
    initial_amount: float,
    dates=None,
    save_path=None,
    algorithm_name=None,
    stock_name=None,
    cost_pct=1e-3,  
    compare_with_market_strategies=True,
    actual_return=None,
    trade_points=None
) -> Dict:
    """
    Analyze trading results and create visual reports
    
    Args:
        prices: List of stock prices
        actions: List of actions (from -1 to 1)
        portfolio_values: List of portfolio values
        initial_amount: Initial investment amount
        dates: List of dates (optional)
        save_path: Path to save results (optional)
        algorithm_name: Name of the algorithm used
        stock_name: Name of the stock
        cost_pct: Transaction fee percentage (default: 0.1%)
        compare_with_market_strategies: Whether to compare with market strategies
        actual_return: Actual return if already calculated
        trade_points: Actual trade points data from environment
        
    Returns:
        Dict: Trading statistics
    """
    # Create DataFrame
    if dates is None:
        dates = pd.date_range(start='2020-01-01', periods=len(prices))
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Action': actions,
        'Portfolio': portfolio_values
    })
    df['Action_Shift'] = df['Action'].shift(1).fillna(0)
    
    # Calculate returns
    # Sử dụng actual_return nếu được cung cấp
    if actual_return is not None:
        model_return = actual_return
    else:
        # Sử dụng cách tính cũ nếu không được cung cấp
        model_return = (portfolio_values[-1] / initial_amount - 1) * 100
    
    # Calculate Buy & Hold with transaction fee
    first_price = prices[0]
    last_price = prices[-1]
    # Apply transaction fee (one-time buy)
    effective_first_price = first_price * (1 + cost_pct)
    buy_hold_return = (last_price / effective_first_price - 1) * 100
    
    alpha = model_return - buy_hold_return
    buy_hold_values = []
    # Xác định số cổ phiếu mua được vào đầu kỳ (có tính phí giao dịch)
    initial_shares = initial_amount / (first_price * (1 + cost_pct))
    for price in prices:
        buy_hold_values.append(initial_shares * price)

    # Tính các chỉ số từ buy_hold_values
    buy_hold_daily_returns = pd.Series(buy_hold_values).pct_change().dropna()
    buy_hold_sharpe = calculate_sharpe_ratio(buy_hold_daily_returns)
    buy_hold_sortino = calculate_sortino_ratio(buy_hold_daily_returns)
    buy_hold_max_drawdown = ((pd.Series(buy_hold_values).cummax() - pd.Series(buy_hold_values)) / pd.Series(buy_hold_values).cummax()).max() * 100
    
    # Nếu có thông tin giao dịch được cung cấp từ run.py, sử dụng nó
    if trade_points is not None:
        df['Trade_Type'] = 'NO TRADE'
        df['Trade_Size'] = 'NO_TRADE'
        
        for trade in trade_points:
            index = trade['day']
            if 0 <= index < len(df):
                df.loc[index, 'Trade_Type'] = trade['type']
                amount = trade['shares']
                if amount > 0:
                    if amount > 5:  # Có thể điều chỉnh ngưỡng này
                        df.loc[index, 'Trade_Size'] = 'LARGE'
                    else:
                        df.loc[index, 'Trade_Size'] = 'NORMAL'
        
        # Đánh dấu các điểm giao dịch
        df['Trade_Signal'] = df['Trade_Type'] != 'NO TRADE'
        total_trades = df['Trade_Signal'].sum()
    else:
        # Nếu không có thông tin giao dịch, dựa vào ngưỡng hành động theo môi trường
        df['Trade_Signal'] = ((df['Action'] > 0.1) & (df['Action_Shift'] <= 0.1)) | \
                        ((df['Action'] < -0.1) & (df['Action_Shift'] >= -0.1))
        total_trades = df['Trade_Signal'].sum()
        
        # Phân loại giao dịch dựa trên action
        df['Trade_Size'] = 'NO_TRADE'
        df.loc[(df['Action'] > 0.1) & (df['Action'] <= 0.5), 'Trade_Size'] = 'NORMAL'
        df.loc[(df['Action'] < -0.1) & (df['Action'] >= -0.5), 'Trade_Size'] = 'NORMAL'
        df.loc[df['Action'] > 0.5, 'Trade_Size'] = 'LARGE'
        df.loc[df['Action'] < -0.5, 'Trade_Size'] = 'LARGE'
        
        # Phân loại loại giao dịch
        df['Trade_Type'] = 'NO TRADE'
        df.loc[(df['Action'] > 0.1) & (df['Action_Shift'] <= 0.1), 'Trade_Type'] = 'BUY'
        df.loc[(df['Action'] < -0.1) & (df['Action_Shift'] >= -0.1), 'Trade_Type'] = 'SELL'
    
    # Thêm phân loại Position cho biểu đồ
    df['Position'] = 'HOLD'  # Mặc định là HOLD
    df.loc[df['Action'] > 0.5, 'Position'] = 'STRONG BUY'
    df.loc[(df['Action'] > 0.1) & (df['Action'] <= 0.5), 'Position'] = 'BUY'
    df.loc[(df['Action'] < -0.1) & (df['Action'] >= -0.5), 'Position'] = 'SELL'
    df.loc[df['Action'] < -0.5, 'Position'] = 'STRONG SELL'
    
    # Calculate daily returns
    df['Daily_Return'] = df['Portfolio'].pct_change().fillna(0)
    
    # Performance metrics
    sharpe_ratio = np.mean(df['Daily_Return']) / np.std(df['Daily_Return']) * np.sqrt(252) if np.std(df['Daily_Return']) > 0 else 0
    # Thêm sortino ratio
    sortino_ratio = calculate_sortino_ratio(df['Daily_Return'])
    max_drawdown = (df['Portfolio'].cummax() - df['Portfolio']).max() / df['Portfolio'].cummax().max()
    
    # Create detailed trade table
    trade_days = df[df['Trade_Type'] != 'NO TRADE'].copy()
    
    # Create charts and save data
    if save_path:
        create_trading_charts(df, initial_amount, model_return, buy_hold_return, save_path, stock_name, algorithm_name)
        df.to_csv(f"{save_path}/trading_results.csv", index=False)
        
        # Save detailed trading table if we have any trades
        if not trade_days.empty:
            trade_days.to_csv(f"{save_path}/trade_details.csv", index=False)
            
            # Print trading details to terminal
            print("\nTrading List:")
            print(f"{'Date':12} | {'Type':10} | {'Price':8} | {'Position':6} | {'Portfolio':12}")
            print('-' * 55)
            for idx, row in trade_days.iterrows():
                print(f"{row['Date'].strftime('%Y-%m-%d'):12} | {row['Trade_Type']:10} | {row['Price']:8.2f} | {row['Action']:6.2f} | {row['Portfolio']:12.2f}")
    
    # Statistics summary
    statistics = {
        'total_trades': int(total_trades),
        'model_return': model_return,
        'buy_hold_return': buy_hold_return,
        'alpha': alpha,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown * 100,
        'final_portfolio': portfolio_values[-1]
    }
    
    # Thêm phần so sánh với chiến lược thị trường
    if compare_with_market_strategies:
        market_results = implement_market_strategies(
            prices=prices, 
            dates=dates,
            initial_amount=initial_amount,
            cost_pct=cost_pct
        )
        
        # Thêm thông tin so sánh vào statistics
        statistics['market_ma_return'] = market_results['MA']['returns'] 
        statistics['market_ma_sharpe'] = market_results['MA']['sharpe_ratio']
        statistics['market_ma_sortino'] = market_results['MA']['sortino_ratio']
        statistics['market_ma_max_drawdown'] = market_results['MA']['max_drawdown']
        statistics['market_ma_trades'] = market_results['MA']['trades']
        
        statistics['market_sr_return'] = market_results['SR']['returns']
        statistics['market_sr_sharpe'] = market_results['SR']['sharpe_ratio']
        statistics['market_sr_sortino'] = market_results['SR']['sortino_ratio']
        statistics['market_sr_max_drawdown'] = market_results['SR']['max_drawdown']
        statistics['market_sr_trades'] = market_results['SR']['trades']
        statistics['buy_hold_sharpe'] = buy_hold_sharpe
        statistics['buy_hold_sortino'] = buy_hold_sortino
        statistics['buy_hold_max_drawdown'] = buy_hold_max_drawdown

        # # In kết quả so sánh
        # if save_path:
        #     print("\n==== Strategy Comparison ====")
        #     print(f"{'Strategy':20} | {'Return %':10} | {'Sharpe':10} | {'Sortino':10} | {'Max DD %':10} | {'Trades':10}")
        #     print("-" * 80)
        #     print(f"{algorithm_name:20} | {statistics['model_return']:10.2f} | {statistics['sharpe_ratio']:10.2f} | {statistics['sortino_ratio']:10.2f} | {statistics['max_drawdown']:10.2f} | {statistics['total_trades']:10}")
        #     print(f"{'Moving Average':20} | {statistics['market_ma_return']:10.2f} | {statistics['market_ma_sharpe']:10.2f} | {statistics['market_ma_sortino']:10.2f} | {statistics['market_ma_max_drawdown']:10.2f} | {statistics['market_ma_trades']:10.0f}")
        #     print(f"{'Signal Rolling':20} | {statistics['market_sr_return']:10.2f} | {statistics['market_sr_sharpe']:10.2f} | {statistics['market_sr_sortino']:10.2f} | {statistics['market_sr_max_drawdown']:10.2f} | {statistics['market_sr_trades']:10.0f}")
        #     print(f"{'Buy & Hold':20} | {statistics['buy_hold_return']:10.2f} | {statistics['buy_hold_sharpe']:10.2f} | {statistics['buy_hold_sortino']:10.2f} | {statistics['buy_hold_max_drawdown']:10.2f} | {'1':10}")
        #     print("-" * 80)
    
    return statistics


def create_trading_charts(df, initial_amount, model_return, buy_hold_return, save_path, stock_name, algorithm_name="DDPG"):
    """Create trading analysis charts"""
    plt.style.use('ggplot')  # Use nicer style
    
    # Đảm bảo có cột 'Position' trước khi groupby
    if 'Position' not in df.columns:
        df['Position'] = 'HOLD'  # Mặc định tất cả là HOLD nếu không có phân loại
        df.loc[df['Action'] > 0.5, 'Position'] = 'STRONG BUY'
        df.loc[(df['Action'] > 0.1) & (df['Action'] <= 0.5), 'Position'] = 'BUY'
        df.loc[(df['Action'] < -0.1) & (df['Action'] >= -0.5), 'Position'] = 'SELL'
        df.loc[df['Action'] < -0.5, 'Position'] = 'STRONG SELL'
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
    
    # Price chart
    ax1.plot(df['Date'], df['Price'], label=f'{stock_name} Price', linewidth=2, color='#1f77b4')
    ax1.set_ylabel('Price (VND)', fontweight='bold')
    ax1.set_title(f'{stock_name} Stock Price', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trade markers on price chart
    # Phân loại điểm giao dịch theo kích thước
    normal_buy_points = df[(df['Trade_Type'] == 'BUY') & (df['Trade_Size'] == 'NORMAL')]
    normal_sell_points = df[(df['Trade_Type'] == 'SELL') & (df['Trade_Size'] == 'NORMAL')]
    large_buy_points = df[(df['Trade_Type'] == 'BUY') & (df['Trade_Size'] == 'LARGE')]
    large_sell_points = df[(df['Trade_Type'] == 'SELL') & (df['Trade_Size'] == 'LARGE')]

    # Hiển thị giao dịch normal
    if not normal_buy_points.empty:
        ax1.scatter(normal_buy_points['Date'], normal_buy_points['Price'], color='green', marker='^', s=80, label='Buy Signal')
    if not normal_sell_points.empty:
        ax1.scatter(normal_sell_points['Date'], normal_sell_points['Price'], color='red', marker='v', s=80, label='Sell Signal')

    # Hiển thị giao dịch large với kích thước lớn hơn
    if not large_buy_points.empty:
        ax1.scatter(large_buy_points['Date'], large_buy_points['Price'], color='darkgreen', marker='^', s=120, label='Large Buy Signal')
    if not large_sell_points.empty:
        ax1.scatter(large_sell_points['Date'], large_sell_points['Price'], color='darkred', marker='v', s=120, label='Large Sell Signal')
    
    # Chỉ thêm legend nếu có các giao dịch
    if (not normal_buy_points.empty or not normal_sell_points.empty or 
        not large_buy_points.empty or not large_sell_points.empty):
        ax1.legend(loc='upper left')
    
    # Action chart
    ax2.plot(df['Date'], df['Action'], label='Action', color='green', linewidth=1.5)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    ax2.axhline(y=-0.5, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.1, color='lightgreen', linestyle=':', alpha=0.3)
    ax2.axhline(y=-0.1, color='lightcoral', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Action (-1 to 1)', fontweight='bold')
    ax2.set_title('Agent Actions', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add color zones to represent different actions
    ax2.fill_between(df['Date'], 0.5, 1, color='green', alpha=0.2, label='Strong Buy Zone')
    ax2.fill_between(df['Date'], 0.1, 0.5, color='lightgreen', alpha=0.2, label='Buy Zone')
    ax2.fill_between(df['Date'], -0.1, 0.1, color='gray', alpha=0.2, label='Hold Zone')
    ax2.fill_between(df['Date'], -0.5, -0.1, color='lightcoral', alpha=0.2, label='Sell Zone')
    ax2.fill_between(df['Date'], -1, -0.5, color='red', alpha=0.2, label='Strong Sell Zone')
    
    # Biểu đồ hành động
    # Thêm điểm giao dịch với kích thước khác nhau
    for i, action in enumerate(df['Action']):
        if df['Trade_Type'].iloc[i] == 'BUY':
            if df['Trade_Size'].iloc[i] == 'LARGE':
                ax2.scatter(df['Date'].iloc[i], action, color='darkgreen', marker='^', s=120, zorder=5)
            elif df['Trade_Size'].iloc[i] == 'NORMAL':
                ax2.scatter(df['Date'].iloc[i], action, color='green', marker='^', s=80, zorder=5)
        elif df['Trade_Type'].iloc[i] == 'SELL':
            if df['Trade_Size'].iloc[i] == 'LARGE':
                ax2.scatter(df['Date'].iloc[i], action, color='darkred', marker='v', s=120, zorder=5)
            elif df['Trade_Size'].iloc[i] == 'NORMAL':
                ax2.scatter(df['Date'].iloc[i], action, color='red', marker='v', s=80, zorder=5)

    ax2.legend(loc='upper left')

    # Biểu đồ giá trị danh mục
    # Hiển thị điểm giao dịch với kích thước khác nhau
    for i, p_value in enumerate(df['Portfolio']):
        if df['Trade_Type'].iloc[i] == 'BUY':
            if df['Trade_Size'].iloc[i] == 'LARGE':
                ax3.scatter(df['Date'].iloc[i], p_value, color='darkgreen', marker='^', s=120, zorder=5)
            elif df['Trade_Size'].iloc[i] == 'NORMAL':
                ax3.scatter(df['Date'].iloc[i], p_value, color='green', marker='^', s=80, zorder=5)
        elif df['Trade_Type'].iloc[i] == 'SELL':
            if df['Trade_Size'].iloc[i] == 'LARGE':
                ax3.scatter(df['Date'].iloc[i], p_value, color='darkred', marker='v', s=120, zorder=5)
            elif df['Trade_Size'].iloc[i] == 'NORMAL':
                ax3.scatter(df['Date'].iloc[i], p_value, color='red', marker='v', s=80, zorder=5)
    ax3.set_ylabel('Portfolio Value', fontweight='bold')
    ax3.set_title(f'Portfolio Comparison ({algorithm_name}: {model_return:.2f}% vs Buy & Hold: {buy_hold_return:.2f}%)', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax3.plot(df['Date'], df['Portfolio'], label='Portfolio Value', color='blue', linewidth=2)

    # Đường Buy & Hold để so sánh
    buy_hold_values = [initial_amount * (price / df['Price'].iloc[0]) for price in df['Price']]
    ax3.plot(df['Date'], buy_hold_values, label='Buy & Hold', color='green', linewidth=1.5, linestyle='--')
    ax3.legend(loc='upper left')
    
    # Format x-axis
    plt.xticks(rotation=45)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/trading_analysis.png", dpi=300)
    plt.close('all')  # Close figures to free memory
    
    # Create return distribution chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df['Daily_Return'].hist(bins=50, ax=ax)
    ax.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{save_path}/return_distribution.png", dpi=300)
    plt.close('all')  # Close figures to free memory
    
    # Tạo biểu đồ phân tích vị trí theo thời gian
    try:
        fig, ax = plt.subplots(figsize=(15, 8))
        colors = {
            'STRONG BUY': 'darkgreen', 
            'BUY': 'green', 
            'HOLD': 'gray', 
            'SELL': 'red', 
            'STRONG SELL': 'darkred'
        }
        
        # Kiểm tra và tạo nhãn cho legend
        legend_handles = []
        legend_labels = []
        
        for position, color in colors.items():
            position_data = df[df['Position'] == position]
            if not position_data.empty:
                scatter = ax.scatter(position_data['Date'], position_data['Price'], 
                                     s=30, color=color, label=position)
                legend_handles.append(scatter)
                legend_labels.append(position)
        
        # Vẽ đường giá cơ bản
        ax.plot(df['Date'], df['Price'], '-', color='blue', alpha=0.3)
        ax.set_title('Position Analysis Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Chỉ thêm legend nếu có dữ liệu
        if legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels)
            
        plt.tight_layout()
        plt.savefig(f"{save_path}/position_analysis.png", dpi=300)
        plt.close('all')  # Close figures to free memory
    except Exception as e:
        print(f"Warning: Could not create position analysis chart: {e}")


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """Tính toán Sharpe ratio"""
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    if len(returns) < 2:
        return 0.0
    
    # Tính lợi nhuận nếu input là giá trị danh mục
    if not all(-1 <= r <= 1 for r in returns[:10]):  # Kiểm tra nếu là giá trị thay vì phần trăm
        returns = returns.pct_change().dropna()
    
    excess_returns = returns - risk_free_rate / periods
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods) if excess_returns.std() > 0 else 0

def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods=252):
    """Tính toán Sortino ratio (chỉ xét đến rủi ro giảm giá)"""
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    if len(returns) < 2:
        return 0.0
    
    # Tính lợi nhuận nếu input là giá trị danh mục
    if not all(-1 <= r <= 1 for r in returns[:10]):  # Kiểm tra nếu là giá trị thay vì phần trăm
        returns = returns.pct_change().dropna()
    
    excess_returns = returns.mean() - risk_free_rate / periods
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    return excess_returns / downside_std * np.sqrt(periods) if downside_std > 0 else 0


def implement_market_strategies(prices, dates, initial_amount=1e6, cost_pct=1e-3):
    """Triển khai các chiến lược thị trường truyền thống"""
    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
 
    # 1. Chiến lược Moving Average
    ma_short, ma_long = 5, 20
    df[f'MA_{ma_short}'] = df['Price'].rolling(window=ma_short, min_periods=1).mean()
    df[f'MA_{ma_long}'] = df['Price'].rolling(window=ma_long, min_periods=1).mean()
    
    # Tạo tín hiệu: 1 khi MA ngắn > MA dài (mua), 0 khi MA ngắn < MA dài (bán)
    df['Signal_MA'] = 0
    df.loc[df[f'MA_{ma_short}'] > df[f'MA_{ma_long}'], 'Signal_MA'] = 1
    df['Position_Change_MA'] = df['Signal_MA'].diff()
    
    # Tính portfolio
    df['MA_Holdings'] = 0.0
    df['MA_Cash'] = initial_amount
    df['MA_Portfolio'] = initial_amount
    
    for i in range(1, len(df)):
        prev_holdings = df['MA_Holdings'].iloc[i-1]
        prev_cash = df['MA_Cash'].iloc[i-1]
        price = df['Price'].iloc[i]
        
        # Mua khi tín hiệu thay đổi từ 0 lên 1
        if df['Position_Change_MA'].iloc[i] == 1:
            shares_to_buy = prev_cash / (price * (1 + cost_pct))
            df.loc[df.index[i], 'MA_Holdings'] = shares_to_buy
            df.loc[df.index[i], 'MA_Cash'] = 0
        # Bán khi tín hiệu thay đổi từ 1 xuống 0
        elif df['Position_Change_MA'].iloc[i] == -1:
            cash_from_sale = prev_holdings * price * (1 - cost_pct)
            df.loc[df.index[i], 'MA_Holdings'] = 0
            df.loc[df.index[i], 'MA_Cash'] = prev_cash + cash_from_sale
        else:
            df.loc[df.index[i], 'MA_Holdings'] = prev_holdings
            df.loc[df.index[i], 'MA_Cash'] = prev_cash
        
        df.loc[df.index[i], 'MA_Portfolio'] = df['MA_Holdings'].iloc[i] * price + df['MA_Cash'].iloc[i]
    
    # 2. Chiến lược Signal Rolling (Bollinger Bands)
    window = 20
    num_std = 2
    df['Rolling_Mean'] = df['Price'].rolling(window=window, min_periods=1).mean()
    df['Rolling_Std'] = df['Price'].rolling(window=window, min_periods=1).std()
    df['Upper_Band'] = df['Rolling_Mean'] + (num_std * df['Rolling_Std'])
    df['Lower_Band'] = df['Rolling_Mean'] - (num_std * df['Rolling_Std'])
    
    df['Signal_SR'] = 0
    df.loc[df['Price'] <= df['Lower_Band'], 'Signal_SR'] = 1  # Mua
    df.loc[df['Price'] >= df['Upper_Band'], 'Signal_SR'] = -1  # Bán
    
    df['SR_Holdings'] = 0.0
    df['SR_Cash'] = initial_amount
    df['SR_Portfolio'] = initial_amount
    
    position = 0
    for i in range(1, len(df)):
        prev_holdings = df['SR_Holdings'].iloc[i-1]
        prev_cash = df['SR_Cash'].iloc[i-1]
        price = df['Price'].iloc[i]
        
        if df['Signal_SR'].iloc[i] == 1 and position == 0:
            shares_to_buy = prev_cash / (price * (1 + cost_pct))
            df.loc[df.index[i], 'SR_Holdings'] = shares_to_buy
            df.loc[df.index[i], 'SR_Cash'] = 0
            position = 1
        elif df['Signal_SR'].iloc[i] == -1 and position == 1:
            cash_from_sale = prev_holdings * price * (1 - cost_pct)
            df.loc[df.index[i], 'SR_Holdings'] = 0
            df.loc[df.index[i], 'SR_Cash'] = prev_cash + cash_from_sale
            position = 0
        else:
            df.loc[df.index[i], 'SR_Holdings'] = prev_holdings
            df.loc[df.index[i], 'SR_Cash'] = prev_cash
        
        df.loc[df.index[i], 'SR_Portfolio'] = df['SR_Holdings'].iloc[i] * price + df['SR_Cash'].iloc[i]
    
    # Tính kết quả
    ma_returns = (df['MA_Portfolio'].iloc[-1] / initial_amount - 1) * 100
    sr_returns = (df['SR_Portfolio'].iloc[-1] / initial_amount - 1) * 100
    
    ma_sharpe = calculate_sharpe_ratio(df['MA_Portfolio'].pct_change().dropna())
    sr_sharpe = calculate_sharpe_ratio(df['SR_Portfolio'].pct_change().dropna())
    
    ma_sortino = calculate_sortino_ratio(df['MA_Portfolio'].pct_change().dropna())
    sr_sortino = calculate_sortino_ratio(df['SR_Portfolio'].pct_change().dropna())
    
    ma_max_drawdown = ((df['MA_Portfolio'].cummax() - df['MA_Portfolio']) / df['MA_Portfolio'].cummax()).max() * 100
    sr_max_drawdown = ((df['SR_Portfolio'].cummax() - df['SR_Portfolio']) / df['SR_Portfolio'].cummax()).max() * 100
    
    ma_trades = abs(df['Position_Change_MA']).sum()
    sr_trades = (df['Signal_SR'] != 0).sum()
    
    results = {
        'MA': {
            'returns': ma_returns,
            'sharpe_ratio': ma_sharpe,
            'sortino_ratio': ma_sortino,
            'max_drawdown': ma_max_drawdown,
            'trades': ma_trades
        },
        'SR': {
            'returns': sr_returns,
            'sharpe_ratio': sr_sharpe,
            'sortino_ratio': sr_sortino,
            'max_drawdown': sr_max_drawdown,
            'trades': sr_trades
        }
    }
    
    return results