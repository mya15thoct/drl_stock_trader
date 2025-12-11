import os
import torch as th
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import argparse

from train.config import Config, build_env
from train.replay_buffer import ReplayBuffer
from train.evaluator import Evaluator
from agents.agent_ddpg import AgentDDPG
from envs.stock_env import StockTradingEnv
from utils.visualize import analyze_trading_results 
from envs.stock_env import get_stock_codes_from_directory


def train_agent_single_process(args: Config) -> Dict[str, Any]:
    """
    Huấn luyện agent với môi trường giao dịch cổ phiếu 
    
    Args:
        args: Cấu hình huấn luyện
        
    Returns:
        Dict: Thông tin huấn luyện và kết quả
    """
    # Khởi tạo cấu hình
    args.init_before_training()
    th.set_grad_enabled(False)
    
    th.backends.cudnn.benchmark = True
    # Khởi tạo môi trường
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    
    # Khởi tạo agent
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)  # Tải model nếu có
    
    # Khởi tạo trạng thái ban đầu
    state, info_dict = env.reset()
    if args.num_envs == 1:
        assert state.shape == (args.state_dim,)
        assert isinstance(state, np.ndarray)
        state = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
    else:
        state = state.to(agent.device)
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, th.Tensor)
    agent.last_state = state.detach()
    
    # Khởi tạo buffer (DDPG is off-policy)
    buffer = ReplayBuffer(
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
        gpu_id=args.gpu_id,
        if_use_per=args.if_use_per,
    )
        
    # Khởi tạo evaluator
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env_args = eval_env_args.copy()
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)
    
    # Biến vòng lặp huấn luyện
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_save_buffer = args.if_save_buffer

    # Tạo và cập nhật buffer ban đầu để đảm bảo đủ dữ liệu
    print("Filling initial buffer...")
    with th.no_grad():
        for _ in range(max(4, int(args.buffer_init_size // horizon_len))):
            buffer_items = agent.explore_env(env, horizon_len)
            buffer.update(buffer_items)
    print(f"Initial buffer size: {buffer.cur_size}")

    
    if_train = True
    while if_train:
        # Khám phá môi trường
        buffer_items = agent.explore_env(env, horizon_len)
        buffer.update(buffer_items)
        
        # Hiển thị thông tin hành động
        actions = buffer_items[1].cpu().numpy()
        action_mean = np.mean(actions)
        action_std = np.std(actions)
        explore_rate = agent.explore_noise_std
        reward_mean = buffer_items[2].mean().item()
        
        # Huấn luyện agent
        th.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        th.set_grad_enabled(False)        
        
        # Đánh giá và lưu thông tin
        evaluator.evaluate_and_save(
            actor=agent.act, 
            steps=horizon_len, 
            exp_r=reward_mean, 
            logging_tuple=(*logging_tuple, explore_rate, f"a_m{action_mean:+.2f}")
        )
        
        # Kiểm tra điều kiện dừng
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
 
    # Kết thúc huấn luyện
    print(f'| usage time: {time.time() - evaluator.start_time:>7.0f} | Saved folder: {cwd}', flush=True)
    
    env.close() if hasattr(env, 'close') else None
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
        buffer.save_or_load_history(cwd, if_save=True)
    
    # Kết quả huấn luyện
    result_dict = {
        'env_name': args.env_name,
        'cwd': cwd,
        'agent_class': agent.__class__.__name__,
        'max_reward': evaluator.max_r,
        'total_step': evaluator.total_step,
        'speed': evaluator.total_step / (time.time() - evaluator.start_time),
    }
    return result_dict


def train_trading_ddpg(stock_code, data_path=None):
    """
    Huấn luyện mô hình DDPG cho giao dịch cổ phiếu
    """
    # Xác định đường dẫn dữ liệu nếu không được cung cấp
    if data_path is None:
        data_path = f'./trainingset/{stock_code}.csv'
    
    # Tạo môi trường tạm thời để xác định loại cổ phiếu
    test_env = StockTradingEnv(data_path=data_path)
    test_env.reset()
    stock_type = test_env.stock_type
    
    # Cấu hình môi trường
    env_args = {
        'data_path': data_path,
        'env_name': f'{stock_code}StockTrading',
        'num_envs': 1,
        'max_step': 500,
        'state_dim': test_env.state_dim,
        'action_dim': test_env.action_dim,
        'if_discrete': False,
        'train_test_split': 0.8,
        'use_train': True,
        'initial_amount': 1e6,
        'max_stock': 1e2,
        'cost_pct': 0.001
    }
    
    # Cấu hình agent
    agent_class = AgentDDPG
    args = Config(agent_class=agent_class, env_class=StockTradingEnv, env_args=env_args)
    
    print(f"\nStock: {stock_code} | Type: {stock_type}")
    
    # Thiết lập tham số dựa trên loại cổ phiếu
    if stock_type == 'High_Risk':
        # Tham số cho cổ phiếu High_Risk
        args.reward_scale = 2 ** -5
        args.explore_noise_std = 0.25
        args.learning_rate = 8e-5
        args.batch_size = 1024
        args.soft_update_tau = 0.003
        args.buffer_size = int(8e5)
        args.repeat_times = 1.0
        args.horizon_len = 2048
        args.net_dims = [512, 384, 256]
        args.gamma = 0.98
        args.clip_grad_norm = 3.0
        print(f"Using High_Risk parameters")
        
    elif stock_type == 'Medium_Risk':
        # Tham số cho cổ phiếu Medium_Risk
        args.reward_scale = 2 ** -6
        args.explore_noise_std = 0.18
        args.learning_rate = 1e-4
        args.batch_size = 512
        args.soft_update_tau = 0.005
        args.buffer_size = int(6e5)
        args.repeat_times = 1.0
        args.horizon_len = 1536
        args.net_dims = [384, 256, 128]
        args.gamma = 0.98
        args.clip_grad_norm = 2.0
        print(f"Using Medium_Risk parameters")
        
    else:  # Low_Risk
        # Tham số cho cổ phiếu Low_Risk
        args.reward_scale = 2 ** -7
        args.explore_noise_std = 0.12
        args.learning_rate = 6e-5
        args.batch_size = 256
        args.soft_update_tau = 0.008
        args.buffer_size = int(4e5)
        args.repeat_times = 0.8
        args.horizon_len = 1024
        args.net_dims = [256, 128]
        args.gamma = 0.99
        args.clip_grad_norm = 1.5
        print(f"Using Low_Risk parameters")
    
    # Tham số đánh giá chung
    args.buffer_init_size = args.batch_size * 20
    args.eval_per_step = int(2e4)
    args.break_step = int(5e5)
    args.gpu_id = 0
    args.eval_times = 4
    
    # Bắt đầu huấn luyện
    result_dict = train_agent_single_process(args)
    print(f"Training results: {result_dict}")
    return result_dict


def test_trained_ddpg_agent(agent_path, test_episodes=10, stock_code=None, data_path=None):
    """
    Thử nghiệm agent DDPG đã được huấn luyện
    
    Args:
        agent_path: Đường dẫn đến thư mục chứa mô hình đã huấn luyện
        test_episodes: Số episode để kiểm tra
        stock_code: Mã cổ phiếu
        data_path: Đường dẫn dữ liệu
    """
    # Xác định đường dẫn dữ liệu nếu không được cung cấp
    if data_path is None:
        data_path = f'./trainingset/{stock_code}.csv'
        
    # Cấu hình môi trường
    env_args = {
        'data_path': data_path,
        'env_name': f'{stock_code}StockTrading',
        'num_envs': 1,
        'max_step': 500,
        'state_dim': 11,
        'action_dim': 1,
        'if_discrete': False,
        'train_test_split': 0.8,  
        'use_train': False,  # Sử dụng tập kiểm thử
    }
    
    # Khởi tạo môi trường và agent
    env = StockTradingEnv(data_path=env_args['data_path'], 
                        train_test_split=env_args['train_test_split'], 
                        use_train=env_args['use_train'])
    env_args['state_dim'] = env.state_dim
    env_args['action_dim'] = env.action_dim
    env_args['max_step'] = env.max_step
    
    # Đường dẫn đến file mô hình actor
    actor_path = f"{agent_path}/actor.pth"
    if not os.path.exists(actor_path):
        print(f"Model file not found: {actor_path}")
        return
    
    # Khởi tạo agent 
    agent = AgentDDPG(net_dims=[128, 64], state_dim=env.state_dim, action_dim=env.action_dim, gpu_id=0)
    
    # Tải mô hình actor
    loaded_actor = th.load(actor_path, map_location=agent.device, weights_only=False)
    agent.act = loaded_actor
    
    # Thử nghiệm
    print(f"\n{'='*60}")
    print(f"Testing DDPG agent for {stock_code}")
    print(f"{'='*60}")
    
    total_returns = []
    all_episodes_data = []
    
    for episode in range(test_episodes):
        env.if_random_reset = False
        state, _ = env.reset()
        
        done = False
        episode_return = 0
        episode_steps = 0
        
        # Data for episode
        actions_list = []
        prices_list = []
        portfolio_values = []
        dates_list = []
        all_trades = []
        total_fees = 0
        last_position = 0

        while not done:
            # Lấy hành động từ agent
            state_tensor = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
            action = agent.get_action(state_tensor, if_deterministic=True)
            
            # Save info
            actions_list.append(action[0])
            prices_list.append(env.close_ary[env.day, 0])
            portfolio_values.append(env.total_asset)
            dates_list.append(env.dates[env.day])

            # Thực hiện hành động
            next_state, reward, terminal, truncated, _ = env.step(action)
            
            # Theo dõi giao dịch
            current_price = env.close_ary[env.day, 0]
            current_position = env.shares[0]
            position_change = current_position - last_position
            
            if position_change != 0:  # Có giao dịch xảy ra
                trade_type = "BUY" if position_change > 0 else "SELL"
                trade_amount = abs(position_change * current_price)
                trade_fee = trade_amount * env.cost_pct
                total_fees += trade_fee
                
                all_trades.append({
                    'day': env.day,
                    'date': str(env.dates[env.day]),
                    'price': current_price,
                    'type': trade_type,
                    'shares': abs(position_change),
                    'amount': trade_amount,
                    'fee': trade_fee,
                    'new_position': current_position,
                    'cash': env.amount,
                    'total_asset': env.total_asset
                })
                
                last_position = current_position
            
            # Cập nhật trạng thái và phần thưởng
            state = next_state
            episode_return += reward
            episode_steps += 1
            
            # Kiểm tra kết thúc
            done = terminal or truncated
        
        # Lưu phần thưởng và hiển thị
        total_returns.append(env.cumulative_returns)
        print(f"Episode {episode + 1}: Return= {env.cumulative_returns:.2f}%, Steps= {episode_steps}, Trades= {len(all_trades)}, Fees= ${total_fees:.2f}")

        # Save data of episode
        all_episodes_data.append({
            'returns': env.cumulative_returns,
            'actions': actions_list,
            'prices': prices_list,
            'portfolio_values': portfolio_values,
            'initial_amount': env.initial_amount,
            'dates': dates_list,
            'trades': all_trades,
            'total_fees': total_fees
        })

    # Find data of best episode
    if all_episodes_data:
        best_episode_idx = np.argmax([data['returns'] for data in all_episodes_data])
        best_episode_data = all_episodes_data[best_episode_idx]

        print(f"\n{'='*60}")
        print(f"BEST EPISODE RESULTS (Episode {best_episode_idx+1})")
        print(f"{'='*60}")
        print(f"Return: {best_episode_data['returns']:.2f}%")
        print(f"Total Trades: {len(best_episode_data['trades'])}")
        print(f"Transaction Fees: ${best_episode_data['total_fees']:.2f} ({best_episode_data['total_fees'] / best_episode_data['initial_amount'] * 100:.4f}%)")
        
        # Analyze and visualize trading results
        statistics = analyze_trading_results(
            prices=best_episode_data['prices'],
            actions=[a[0] for a in best_episode_data['actions']],
            portfolio_values=best_episode_data['portfolio_values'],
            initial_amount=best_episode_data['portfolio_values'][0],
            dates=best_episode_data['dates'],
            save_path=agent_path,
            algorithm_name="DDPG",
            stock_name=stock_code,
            cost_pct=env.cost_pct,
            compare_with_market_strategies=True,
            actual_return=best_episode_data['returns'],
            trade_points=best_episode_data['trades']
        )
        
        # Print comparison with other strategies
        print(f"\n{'='*80}")
        print(f"STRATEGY COMPARISON")
        print(f"{'='*80}")
        print(f"{'Strategy':20} | {'Return %':>10} | {'Sharpe':>8} | {'Sortino':>8} | {'Max DD %':>10} | {'Trades':>8}")
        print("-" * 80)
        print(f"{'DDPG':20} | {best_episode_data['returns']:>10.2f} | {statistics['sharpe_ratio']:>8.2f} | {statistics['sortino_ratio']:>8.2f} | {statistics['max_drawdown']:>10.2f} | {len(best_episode_data['trades']):>8}")
        print(f"{'Moving Average':20} | {statistics['market_ma_return']:>10.2f} | {statistics['market_ma_sharpe']:>8.2f} | {statistics['market_ma_sortino']:>8.2f} | {statistics['market_ma_max_drawdown']:>10.2f} | {statistics['market_ma_trades']:>8.0f}")
        print(f"{'Signal Rolling':20} | {statistics['market_sr_return']:>10.2f} | {statistics['market_sr_sharpe']:>8.2f} | {statistics['market_sr_sortino']:>8.2f} | {statistics['market_sr_max_drawdown']:>10.2f} | {statistics['market_sr_trades']:>8.0f}")
        print(f"{'Buy & Hold':20} | {statistics['buy_hold_return']:>10.2f} | {statistics['buy_hold_sharpe']:>8.2f} | {statistics['buy_hold_sortino']:>8.2f} | {statistics['buy_hold_max_drawdown']:>10.2f} | {'1':>8}")
        print("=" * 80)

        print(f"\nChart saved: {agent_path}/trading_analysis.png")
        print(f"Results saved: {agent_path}/trading_results.csv")
        
        # Save all trades
        if any(data['trades'] for data in all_episodes_data):
            import pandas as pd
            all_trades_list = []
            for i, episode_data in enumerate(all_episodes_data):
                for trade in episode_data['trades']:
                    trade['episode'] = i + 1
                    all_trades_list.append(trade)
            
            if all_trades_list:
                all_trades_df = pd.DataFrame(all_trades_list)
                all_trades_df.to_csv(f"{agent_path}/all_detailed_trades.csv", index=False)
                print(f"All trades saved: {agent_path}/all_detailed_trades.csv")

    # Display summary
    if total_returns:
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        print(f"\n{'='*60}")
        print(f"SUMMARY ({test_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Average Return: {avg_return:.2f}% ± {std_return:.2f}%")
        
        if all_episodes_data:
            total_fees_all_episodes = sum(data['total_fees'] for data in all_episodes_data)
            avg_fees = total_fees_all_episodes / len(all_episodes_data)
            print(f"Average Fees per Episode: ${avg_fees:.2f}")


def run_ddpg_for_stocks(data_dir='./us_dataset/', mode='train_and_test', stock_codes=None, test_episodes=5):
    """
    Chạy DDPG agent cho các cổ phiếu
    
    Args:
        data_dir: Thư mục chứa dữ liệu cổ phiếu
        mode: 'train', 'test', hoặc 'train_and_test'
        stock_codes: Danh sách mã cổ phiếu cụ thể để xử lý, nếu None sẽ xử lý tất cả
        test_episodes: Số episode để test
    """
    # Lấy danh sách mã cổ phiếu nếu không được cung cấp
    if stock_codes is None:
        stock_codes = get_stock_codes_from_directory(data_dir)
    
    if not stock_codes:
        print(f"No data found in: {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"DDPG Stock Trading - {len(stock_codes)} stocks")
    print(f"{'='*60}")
    print(f"Stocks: {', '.join(stock_codes)}")
    print(f"Mode: {mode}")
    print(f"Data directory: {data_dir}\n")
    
    # Lặp qua từng mã cổ phiếu
    for idx, stock_code in enumerate(stock_codes, 1):
        data_path = f'{data_dir}/{stock_code}.csv'
        
        print(f"\n{'#'*60}")
        print(f"[{idx}/{len(stock_codes)}] PROCESSING: {stock_code}")
        print(f"{'#'*60}")
        
        # Huấn luyện mô hình
        if mode in ['train', 'train_and_test']:
            print(f"\n--- TRAINING DDPG for {stock_code} ---")
            train_trading_ddpg(stock_code=stock_code, data_path=data_path)

        # Kiểm thử mô hình
        if mode in ['test', 'train_and_test']:
            print(f"\n--- TESTING DDPG for {stock_code} ---")
            model_path = f'./{stock_code}StockTrading_DDPG_0'
            
            if not os.path.exists(model_path):
                print(f"Model folder not found: {model_path}")
                continue
            
            test_trained_ddpg_agent(model_path, test_episodes=test_episodes, stock_code=stock_code, data_path=data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG Stock Trading - Multi-Market Support')
    parser.add_argument('--market', type=str, default='us', choices=['us', 'thailand', 'th'], 
                       help='Market to trade: us or thailand')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='Data directory (optional, auto-selected based on market)')
    parser.add_argument('--mode', type=str, default='train_and_test', 
                       choices=['train', 'test', 'train_and_test'], help='Operation mode')
    parser.add_argument('--stock_code', type=str, default=None, 
                       help='Specific stock code (optional)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Auto-select data directory based on market
    if args.data_dir is None:
        if args.market in ['thailand', 'th']:
            args.data_dir = './thailand_dataset_selected/'
            print(f"🇹🇭 Trading Thailand Market (SET)")
        else:
            args.data_dir = './us_dataset/'
            print(f"🇺🇸 Trading US Market (S&P 500)")
    
    # Determine stocks to process
    if args.stock_code:
        stock_codes = [args.stock_code]
    else:
        stock_codes = None  # Will process all stocks in data_dir
    
    run_ddpg_for_stocks(
        data_dir=args.data_dir,
        mode=args.mode,
        stock_codes=stock_codes,
        test_episodes=args.episodes
    )
