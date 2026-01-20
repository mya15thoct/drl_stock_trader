"""
Script chạy thực nghiệm cho cổ phiếu Việt Nam
- Tải dữ liệu từ CafeF API
- Train và Test model DDPG
- Lưu kết quả tổng hợp vào CSV
"""

import os
import sys
import random
import requests
import pandas as pd
import numpy as np
import torch as th
import time
from datetime import datetime
import argparse
import traceback

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
th.manual_seed(RANDOM_SEED)
if th.cuda.is_available():
    th.cuda.manual_seed(RANDOM_SEED)
    th.cuda.manual_seed_all(RANDOM_SEED)

# Thêm đường dẫn project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import train_trading_ddpg
from envs.stock_env import StockTradingEnv
from agents.agent_ddpg import AgentDDPG
from utils.visualize import analyze_trading_results


# ==================== DANH SÁCH MÃ CỔ PHIẾU VIỆT NAM ====================
# 256 mã từ file medium_risk_analysis_results.csv
VIETNAM_STOCKS = [
    "ABB", "ABI", "ACB", "ACV", "AGG", "AGP", "AGR", "ALT", "AMV", "ANV",
    "ASM", "BAB", "BAF", "BBC", "BCM", "BID", "BMI", "BMP", "BSI", "BTS",
    "BVB", "BVH", "BWE", "CAN", "CEO", "CII", "CJC", "CLL", "CMG", "CMT",
    "CNG", "CPC", "CRC", "CSM", "CSV", "CTD", "CTG", "DBC", "DBD", "DCM",
    "DGC", "DGW", "DHG", "DHT", "DIG", "DP3", "DPM", "DQC", "DRC", "DSE",
    "DXG", "DXS", "EIB", "EVF", "FIT", "FLC", "FMC", "FPT", "FRT", "FTS",
    "GAS", "GDT", "GEG", "GEX", "GMD", "GVR", "HAG", "HAH", "HAP", "HBC",
    "HCM", "HDB", "HDC", "HDG", "HHS", "HHV", "HID", "HII", "HMC", "HNG",
    "HPG", "HPX", "HQC", "HSG", "HT1", "HTI", "HTN", "HTV", "HVN", "IBC",
    "ICG", "IDI", "IJC", "ITA", "ITC", "JVC", "KBC", "KDC", "KDH", "KHG",
    "KLB", "KOS", "KSB", "KSF", "L10", "LAS", "LCG", "LCM", "LDG", "LEC",
    "LGC", "LGL", "LHG", "LIX", "LPB", "LSS", "MBB", "MCG", "MIG", "MSB",
    "MSN", "MWG", "NAF", "NAG", "NAV", "NBB", "NCT", "NHA", "NHH", "NKG",
    "NLG", "NNC", "NT2", "NTL", "NVL", "NVT", "OCB", "OGC", "ORS", "PAC",
    "PAN", "PC1", "PDR", "PET", "PGC", "PGD", "PGI", "PGV", "PHR", "PIT",
    "PJT", "PLX", "PME", "PNC", "PNJ", "POW", "PPC", "PSH", "PTB", "PTC",
    "PTI", "PVD", "PVP", "PVS", "PVT", "QBS", "QCG", "RAL", "REE", "S4A",
    "SAB", "SAM", "SBT", "SBV", "SC5", "SCD", "SCR", "SCS", "SFC", "SFG",
    "SFI", "SGN", "SGR", "SGT", "SHB", "SHI", "SHP", "SHS", "SJD", "SJS",
    "SKG", "SLS", "SMB", "SMC", "SRC", "SRF", "SSB", "SSI", "ST8", "STB",
    "STG", "STK", "SVC", "SVI", "SVT", "SZC", "SZL", "TBC", "TCB", "TCD",
    "TCH", "TCL", "TCM", "TCT", "TDC", "TDH", "TDM", "TDP", "TEG", "THG",
    "TIG", "TIP", "TIX", "TLD", "TLG", "TLH", "TMP", "TMS", "TN1", "TNA",
    "TNC", "TNH", "TNI", "TPB", "TPC", "TRA", "TRC", "TTA", "TTB", "TTF",
    "TVB", "TVS", "VCA", "VCB", "VCF", "VCG", "VCI", "VDP", "VDS", "VGC",
    "VHC", "VHM", "VIB", "VIC", "VIP", "VIX", "VJC", "VMD", "VND", "VNE",
    "VNM", "VNR", "VOS", "VPB", "VPD", "VPG", "VPH", "VPI", "VPS", "VRC",
    "VRE", "VSC", "VSH", "VTB", "VTO"
]


def download_stock_data(stock_code: str, output_dir: str) -> bool:
    """Tải dữ liệu cổ phiếu Việt Nam từ CafeF"""
    try:
        base_url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
        all_data = []
        
        for page_index in range(1, 100):
            params = {
                'Symbol': stock_code,
                'StartDate': '',
                'EndDate': '',
                'PageIndex': page_index,
                'PageSize': 100000
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('Success', False):
                    break
                    
                records = data.get('Data', {}).get('Data', [])
                if not records:
                    break
                    
                all_data.extend(records)
                time.sleep(0.5)
                
            except Exception as e:
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            column_map = {
                'Ngay': 'Date',
                'GiaDongCua': 'close_price',
                'GiaMoCua': 'open_price',
                'GiaCaoNhat': 'high_price',
                'GiaThapNhat': 'low_price',
                'KhoiLuongKhopLenh': 'volume'
            }
            
            available_cols = [col for col in column_map.keys() if col in df.columns]
            if not available_cols:
                print(f"✗ No valid columns for {stock_code}")
                return False
                
            df = df[available_cols].rename(columns=column_map)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.sort_values('Date', ascending=True)
            
            df = df[(df['Date'] >= '2018-01-01') & (df['Date'] < '2025-01-01')]
            
            if len(df) > 100:
                output_path = os.path.join(output_dir, f'{stock_code}.csv')
                df.to_csv(output_path, index=False)
                print(f"✓ Downloaded {stock_code}: {len(df)} rows")
                return True
            else:
                print(f"✗ Not enough data for {stock_code}: {len(df)} rows")
                return False
        else:
            print(f"✗ No data for {stock_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading {stock_code}: {e}")
        return False


def run_experiment(stock_code: str, data_dir: str, results: list, test_episodes: int = 5):
    """Chạy train và test cho một mã cổ phiếu"""
    data_path = os.path.join(data_dir, f'{stock_code}.csv')
    
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        return False
        
    try:
        start_time = time.time()
        
        # Train
        print(f"\n{'='*60}")
        print(f"TRAINING: {stock_code}")
        print(f"{'='*60}")
        train_result = train_trading_ddpg(stock_code=stock_code, data_path=data_path)
        
        # Test
        print(f"\n{'='*60}")
        print(f"TESTING: {stock_code}")
        print(f"{'='*60}")
        model_path = f'./{stock_code}StockTrading_DDPG_0'
        
        if os.path.exists(model_path):
            test_result = run_test_and_get_results(
                agent_path=model_path,
                stock_code=stock_code,
                data_path=data_path,
                test_episodes=test_episodes
            )
            
            elapsed_time = time.time() - start_time
            
            result_entry = {
                'stock_code': stock_code,
                'market': 'Vietnam',
                'train_max_reward': train_result.get('max_reward', 0),
                'train_total_step': train_result.get('total_step', 0),
                'test_avg_return': test_result.get('avg_return', 0),
                'test_std_return': test_result.get('std_return', 0),
                'test_best_return': test_result.get('best_return', 0),
                'test_worst_return': test_result.get('worst_return', 0),
                'sharpe_ratio': test_result.get('sharpe_ratio', 0),
                'sortino_ratio': test_result.get('sortino_ratio', 0),
                'max_drawdown': test_result.get('max_drawdown', 0),
                'buy_hold_return': test_result.get('buy_hold_return', 0),
                'alpha': test_result.get('alpha', 0),
                'total_trades': test_result.get('total_trades', 0),
                'elapsed_time': elapsed_time,
                'random_seed': RANDOM_SEED,
                'status': 'success'
            }
            results.append(result_entry)
            
            print(f"\n✓ {stock_code} completed in {elapsed_time:.1f}s")
            print(f"  Return: {test_result.get('avg_return', 0):.2f}% | Alpha: {test_result.get('alpha', 0):.2f}%")
            return True
        else:
            print(f"✗ Model not found for {stock_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error with {stock_code}: {e}")
        traceback.print_exc()
        results.append({
            'stock_code': stock_code,
            'market': 'Vietnam',
            'status': 'failed',
            'error': str(e)
        })
        return False


def run_test_and_get_results(agent_path: str, stock_code: str, data_path: str, test_episodes: int = 5) -> dict:
    """Chạy test và trả về kết quả chi tiết"""
    env = StockTradingEnv(
        data_path=data_path,
        train_test_split=0.8,
        data_type='test',
        use_train=False
    )
    
    actor_path = f"{agent_path}/actor.pth"
    if not os.path.exists(actor_path):
        return {}
        
    agent = AgentDDPG(net_dims=[128, 64], state_dim=env.state_dim, action_dim=env.action_dim, gpu_id=0)
    loaded_actor = th.load(actor_path, map_location=agent.device, weights_only=False)
    agent.act = loaded_actor
    
    all_returns = []
    all_trades = []
    best_episode_data = None
    best_return = -float('inf')
    
    for episode in range(test_episodes):
        env.if_random_reset = False
        state, _ = env.reset()
        
        done = False
        episode_trades = 0
        actions_list = []
        prices_list = []
        portfolio_values = []
        last_position = 0
        
        while not done:
            state_tensor = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
            action = agent.get_action(state_tensor, if_deterministic=True)
            
            actions_list.append(action[0])
            prices_list.append(env.close_ary[env.day, 0])
            portfolio_values.append(env.total_asset)
            
            next_state, reward, terminal, truncated, _ = env.step(action)
            
            current_position = env.shares[0]
            if current_position != last_position:
                episode_trades += 1
                last_position = current_position
            
            state = next_state
            done = terminal or truncated
        
        all_returns.append(env.cumulative_returns)
        all_trades.append(episode_trades)
        
        if env.cumulative_returns > best_return:
            best_return = env.cumulative_returns
            best_episode_data = {
                'prices': prices_list,
                'actions': [a[0] for a in actions_list],
                'portfolio_values': portfolio_values,
                'initial_amount': env.initial_amount
            }
    
    first_price = env.close_ary[0, 0]
    last_price = env.close_ary[-1, 0]
    buy_hold_return = (last_price / first_price - 1) * 100
    
    statistics = {}
    if best_episode_data:
        statistics = analyze_trading_results(
            prices=best_episode_data['prices'],
            actions=best_episode_data['actions'],
            portfolio_values=best_episode_data['portfolio_values'],
            initial_amount=best_episode_data['initial_amount'],
            save_path=agent_path,
            algorithm_name="DDPG",
            stock_name=stock_code,
            actual_return=best_return
        )
    
    return {
        'avg_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'best_return': max(all_returns),
        'worst_return': min(all_returns),
        'sharpe_ratio': statistics.get('sharpe_ratio', 0),
        'sortino_ratio': statistics.get('sortino_ratio', 0),
        'max_drawdown': statistics.get('max_drawdown', 0),
        'buy_hold_return': buy_hold_return,
        'alpha': np.mean(all_returns) - buy_hold_return,
        'total_trades': np.mean(all_trades)
    }


def save_results(results: list, output_path: str):
    """Lưu kết quả ra file CSV"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    success_results = [r for r in results if r.get('status') == 'success']
    if success_results:
        avg_return = np.mean([r['test_avg_return'] for r in success_results])
        avg_alpha = np.mean([r['alpha'] for r in success_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in success_results])
        
        print(f"\n{'='*60}")
        print(f"VIETNAM EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Completed: {len(success_results)}/{len(results)} stocks")
        print(f"Average Return: {avg_return:.2f}%")
        print(f"Average Alpha:  {avg_alpha:.2f}%")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"{'='*60}")


def run_test_only(stock_code: str, data_dir: str, results: list, test_episodes: int = 5):
    """Chỉ chạy test cho một mã cổ phiếu (không train)"""
    data_path = os.path.join(data_dir, f'{stock_code}.csv')
    model_path = f'./{stock_code}StockTrading_DDPG_0'
    
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return False
        
    try:
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"TESTING: {stock_code}")
        print(f"{'='*60}")
        
        test_result = run_test_and_get_results(
            agent_path=model_path,
            stock_code=stock_code,
            data_path=data_path,
            test_episodes=test_episodes
        )
        
        elapsed_time = time.time() - start_time
        
        result_entry = {
            'stock_code': stock_code,
            'market': 'Vietnam',
            'test_avg_return': test_result.get('avg_return', 0),
            'test_std_return': test_result.get('std_return', 0),
            'test_best_return': test_result.get('best_return', 0),
            'test_worst_return': test_result.get('worst_return', 0),
            'sharpe_ratio': test_result.get('sharpe_ratio', 0),
            'sortino_ratio': test_result.get('sortino_ratio', 0),
            'max_drawdown': test_result.get('max_drawdown', 0),
            'buy_hold_return': test_result.get('buy_hold_return', 0),
            'alpha': test_result.get('alpha', 0),
            'total_trades': test_result.get('total_trades', 0),
            'elapsed_time': elapsed_time,
            'random_seed': RANDOM_SEED,
            'status': 'success'
        }
        results.append(result_entry)
        
        print(f"\n✓ {stock_code} tested in {elapsed_time:.1f}s")
        print(f"  Return: {test_result.get('avg_return', 0):.2f}% | Alpha: {test_result.get('alpha', 0):.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ Error testing {stock_code}: {e}")
        traceback.print_exc()
        results.append({
            'stock_code': stock_code,
            'market': 'Vietnam',
            'status': 'failed',
            'error': str(e)
        })
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Vietnam Stock Experiment')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['all', 'train', 'test'],
                       help='Mode: all (download+train+test), train (train+test), test (test only)')
    parser.add_argument('--data_dir', type=str, default='./data/vietnam',
                       help='Directory to store/read stock data')
    parser.add_argument('--output', type=str, default='./results/vietnam_results.csv',
                       help='Output path for results')
    parser.add_argument('--test_episodes', type=int, default=5,
                       help='Number of test episodes per stock')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start from this index (for resuming)')
    parser.add_argument('--end_index', type=int, default=None,
                       help='End at this index (exclusive)')
    parser.add_argument('--stocks', type=str, default=None,
                       help='Comma-separated list of specific stocks to run')
    
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.stocks:
        stock_list = [s.strip() for s in args.stocks.split(',')]
    else:
        stock_list = VIETNAM_STOCKS
        
    if args.end_index:
        stock_list = stock_list[args.start_index:args.end_index]
    else:
        stock_list = stock_list[args.start_index:]
    
    print(f"\n{'#'*60}")
    print(f"🇻🇳 VIETNAM STOCK EXPERIMENT")
    print(f"{'#'*60}")
    print(f"Total stocks: {len(stock_list)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Mode: {args.mode.upper()}")
    print(f"{'#'*60}\n")
    
    # Download data nếu mode = all
    if args.mode == 'all':
        print("Downloading data from CafeF...")
        for i, stock in enumerate(stock_list, 1):
            print(f"[{i}/{len(stock_list)}] ", end="")
            download_stock_data(stock, args.data_dir)
            time.sleep(1)
        print()
    
    results = []
    for i, stock in enumerate(stock_list, 1):
        print(f"\n[{i}/{len(stock_list)}] Processing {stock}...")
        
        if args.mode == 'test':
            # Chỉ test
            run_test_only(stock, args.data_dir, results, args.test_episodes)
        else:
            # Train + test (mode = all hoặc train)
            run_experiment(stock, args.data_dir, results, args.test_episodes)
        
        if i % 10 == 0:
            save_results(results, args.output.replace('.csv', '_temp.csv'))
    
    save_results(results, args.output)


if __name__ == '__main__':
    main()
