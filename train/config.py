'''

File này thiết lập môi trường giao dịch chứng khoán trong Reinforcement Learning (RL).
Hỗ trợ huấn luyện song song với VecEnv, giúp tăng tốc quá trình học của agent.
Sử dụng multiprocessing.Pipe để giao tiếp giữa các tiến trình con (SubEnv).
Cung cấp phương thức khởi tạo, reset và bước hành động cho môi trường vector hóa.
Cho phép thiết lập và quản lý các thông số môi trường như số bước tối đa, số môi trường con, và loại tác vụ (discrete/continuous).
'''

import os
import numpy as np
import torch as th
from typing import Tuple, Dict
from multiprocessing import Pipe, Process
import sys
sys.path.append('..')
from envs.stock_env import StockTradingEnv
import warnings
from typing import Tuple, Dict, Any
import inspect



TEN = th.Tensor


class Config:
    """
    Lớp cấu hình cho quá trình huấn luyện các agent Deep Reinforcement Learning
    
    :param agent_class: Lớp agent sẽ được sử dụng (AgentDDPG, AgentTD3, AgentSAC...)
    :param env_class: Lớp môi trường (StockTradingEnv)
    :param env_args: Các tham số của môi trường
    """
    
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.num_envs = None  # Số lượng môi trường song song
        self.agent_class = agent_class  # Lớp agent được sử dụng
        self.if_off_policy = self.get_if_off_policy()  # Agent là off-policy hay on-policy

        '''Tham số của môi trường'''
        self.env_class = env_class  # Lớp môi trường
        self.env_args = env_args  # Tham số môi trường
        if env_args is None:  # Tạo tham số giả
            env_args = {'env_name': None,
                        'num_envs': 1,
                        'max_step': 12345,
                        'state_dim': None,
                        'action_dim': None,
                        'if_discrete': None, }
        env_args.setdefault('num_envs', 1)  # Mặc định 1 môi trường
        env_args.setdefault('max_step', 12345)  # Số bước tối đa mặc định
        env_args.setdefault('data_path', None)  # Đường dẫn dữ liệu FPT
        self.env_name = env_args['env_name']  # Tên môi trường, dùng để đặt tên thư mục lưu
        self.num_envs = env_args['num_envs']  # Số lượng môi trường song song
        self.max_step = env_args['max_step']  # Số bước tối đa trong một episode
        self.state_dim = env_args['state_dim']  # Số chiều của không gian trạng thái
        self.action_dim = env_args['action_dim']  # Số chiều của không gian hành động
        self.if_discrete = env_args['if_discrete']  # Không gian hành động rời rạc hay liên tục

        '''Tham số cho điều chỉnh phần thưởng'''
        self.gamma = 0.99  # Hệ số chiết khấu cho phần thưởng tương lai
        self.reward_scale = 2 ** 0  # Hệ số scale phần thưởng

        '''Tham số cho huấn luyện'''
        self.net_dims = [128, 64]  # Kích thước các lớp ẩn của mạng MLP
        self.learning_rate = 1e-4  # Tốc độ học cho cập nhật mạng
        self.clip_grad_norm = 3.0  # Giới hạn cắt gradient sau khi chuẩn hóa
        self.state_value_tau = 0  # Hệ số chuẩn hóa trạng thái và giá trị
        self.soft_update_tau = 5e-3  # Hệ số cập nhật mềm
        if self.if_off_policy:  # Off-policy (DDPG, TD3, SAC...)
            self.batch_size = int(128)  # Kích thước batch lấy từ buffer
            self.horizon_len = int(512)  # Số bước khám phá trước khi cập nhật mạng
            self.buffer_size = int(1e6)  # Kích thước buffer, FIFO cho off-policy
            self.repeat_times = 1.0  # Số lần cập nhật lặp lại
            self.if_use_per = False  # Sử dụng PER (Prioritized Experience Replay)
            self.lambda_fit_cum_r = 0.0  # Critic khớp với giá trị trung bình batch
            self.buffer_init_size = int(self.batch_size * 8)  # Kích thước buffer tối thiểu trước khi huấn luyện
        else:  # On-policy (A2C, PPO...)
            self.batch_size = int(128)  # Kích thước batch lấy từ buffer
            self.horizon_len = int(2048)  # Số bước khám phá trước khi cập nhật mạng
            self.buffer_size = None  # Xóa buffer sau mỗi lần cập nhật
            self.repeat_times = 8.0  # Số lần cập nhật lặp lại
            self.if_use_vtrace = True  # Sử dụng V-trace + GAE
            self.buffer_init_size = None  # Không cần thiết cho on-policy

        '''Tham số cho thiết bị'''
        self.gpu_id = int(0)  # ID của GPU, -1 nghĩa là CPU
        self.num_workers = 2  # Số lượng worker cho mỗi GPU
        self.num_threads = 8  # Số luồng CPU cho pytorch
        self.random_seed = None  # Hạt giống ngẫu nhiên, None có nghĩa là dùng GPU_ID làm hạt
        self.learner_gpu_ids = ()  # ID GPU cho learners, () nghĩa là một GPU hoặc CPU

        '''Tham số đánh giá'''
        self.cwd = None  # Thư mục làm việc hiện tại để lưu mô hình, None nghĩa là tự động đặt
        self.if_remove = True  # Xóa thư mục cwd cũ? (True/False/None: hỏi)
        self.break_step = np.inf  # Dừng huấn luyện nếu 'total_step > break_step'
        self.break_score = np.inf  # Dừng huấn luyện nếu 'cumulative_rewards > break_score'
        self.if_keep_save = True  # Tiếp tục lưu checkpoint, False nghĩa là lưu đến khi dừng
        self.if_over_write = False  # Ghi đè lên mạng policy tốt nhất
        self.if_save_buffer = False  # Lưu replay buffer để tiếp tục huấn luyện sau khi dừng

        self.save_gap = int(8)  # Lưu actor với tên f"{cwd}/actor_*.pth" cho đường cong học tập
        self.eval_times = int(5)  # Số lần đánh giá để lấy trung bình
        self.eval_per_step = int(1e4)  # Đánh giá agent sau mỗi n bước huấn luyện
        self.eval_env_class = None  # Môi trường đánh giá (nếu khác môi trường huấn luyện)
        self.eval_env_args = None  # Tham số cho môi trường đánh giá
        self.eval_record_step = 0  # Bắt đầu ghi lại sau khi khám phá đạt đến bước này

        # Tham số cho khám phá
        self.explore_noise_std = 0.1  # Độ lệch chuẩn của nhiễu khám phá

    def init_before_training(self):
        """Khởi tạo trước khi huấn luyện"""
        if self.random_seed is None:
            self.random_seed = max(0, self.gpu_id)
        np.random.seed(self.random_seed)
        th.manual_seed(self.random_seed)
        th.set_num_threads(self.num_threads)
        th.set_default_dtype(th.float32)

        '''Đặt cwd (thư mục làm việc hiện tại) để lưu mô hình'''
        if self.cwd is None:
            agent_name = self.agent_class.__name__
            if agent_name.startswith('Agent'):
                agent_name = agent_name[5:]  # Bỏ tiền tố 'Agent'
            self.cwd = f'./{self.env_name}_{agent_name}_{self.random_seed}'

        '''Xóa lịch sử'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}", flush=True)
        else:
            print(f"| Arguments Keep cwd: {self.cwd}", flush=True)
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        """Xác định xem agent có phải là off-policy không"""
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])

    def print_config(self):
        """In ra cấu hình hiện tại"""
        from pprint import pprint
        print(pprint(vars(self)), flush=True)  # In các tham số một cách rõ ràng


def build_env(env_class=None, env_args: Dict[str, Any] = None, gpu_id: int = -1):

    """
    Xây dựng môi trường dựa trên lớp và tham số
    
    :param env_class: Lớp môi trường
    :param env_args: Tham số môi trường
    :param gpu_id: ID của GPU
    :return: Môi trường đã được khởi tạo
    """
    warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated.*")
    env_args['gpu_id'] = gpu_id  # Đặt gpu_id cho môi trường vector hóa

    if env_args.get('if_build_vec_env'):
        num_envs = env_args['num_envs']
        env = VecEnv(env_class=env_class, env_args=env_args, num_envs=num_envs, gpu_id=gpu_id)
    elif hasattr(env_class, '__module__') and env_class.__module__ == 'gymnasium.envs.registration':
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))

    env_args.setdefault('num_envs', 1)
    env_args.setdefault('max_step', 12345)

    # Cập nhật các thuộc tính của môi trường
    for attr_str in ('env_name', 'num_envs', 'max_step', 'state_dim', 'action_dim', 'if_discrete'):
        if attr_str in env_args:
            setattr(env, attr_str, env_args[attr_str])
    
    return env


def kwargs_filter(function, kwargs: Dict[str, Any]) -> Dict[str, Any]:

    """
    Lọc kwargs để chỉ giữ lại các tham số phù hợp với hàm
    
    :param function: Hàm cần lọc tham số
    :param kwargs: Từ điển các tham số
    :return: Từ điển các tham số đã lọc
    """
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # Lọc kwargs


"""Môi trường vector hóa"""


class SubEnv(Process):
    """
    Process con để chạy từng môi trường con trong môi trường vector hóa
    """
    def __init__(self, sub_pipe0: Any, vec_pipe1: Any,
                 env_class: Any, env_args: dict, env_id: int = 0):
        super().__init__()
        self.sub_pipe0 = sub_pipe0
        self.vec_pipe1 = vec_pipe1

        self.env_class = env_class
        self.env_args = env_args
        self.env_id = env_id

    def run(self):
        th.set_grad_enabled(False)

        '''Khởi tạo môi trường'''
        if hasattr(self.env_class, '__module__') and self.env_class.__module__ == 'gymnasium.envs.registration':
            env = self.env_class(id=self.env_args['env_name'])
        else:
            env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        '''Đặt hạt ngẫu nhiên cho môi trường'''
        random_seed = self.env_id
        np.random.seed(random_seed)
        th.manual_seed(random_seed)

        while True:
            action = self.sub_pipe0.recv()
            if action is None:
                state, info_dict = env.reset()
                self.vec_pipe1.send((self.env_id, state))
            else:
                state, reward, terminal, truncate, info_dict = env.step(action)

                done = terminal or truncate
                state = env.reset()[0] if done else state
                self.vec_pipe1.send((self.env_id, state, reward, terminal, truncate))


class VecEnv:
    """
    Môi trường vector hóa chạy nhiều môi trường song song
    """
    def __init__(self, env_class: Any, env_args: dict, num_envs: int, gpu_id: int = -1):
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.num_envs = num_envs  # Số lượng môi trường con

        '''Thông tin cần thiết khi thiết kế môi trường tùy chỉnh'''
        self.env_name = env_args['env_name']  # Tên môi trường
        self.max_step = env_args['max_step']  # Số bước tối đa trong một episode
        self.state_dim = env_args['state_dim']  # Số chiều trạng thái
        self.action_dim = env_args['action_dim']  # Số chiều hành động
        self.if_discrete = env_args['if_discrete']  # Hành động rời rạc hay liên tục

        '''Tăng tốc với đa tiến trình: Process, Pipe'''
        assert self.num_envs <= 64
        self.res_list = [[] for _ in range(self.num_envs)]

        sub_pipe0s, sub_pipe1s = list(zip(*[Pipe(duplex=False) for _ in range(self.num_envs)]))
        self.sub_pipe1s = sub_pipe1s

        vec_pipe0, vec_pipe1 = Pipe(duplex=False)  # nhận, gửi
        self.vec_pipe0 = vec_pipe0

        self.sub_envs = [
            SubEnv(sub_pipe0=sub_pipe0, vec_pipe1=vec_pipe1,
                   env_class=env_class, env_args=env_args, env_id=env_id)
            for env_id, sub_pipe0 in enumerate(sub_pipe0s)
        ]

        [setattr(p, 'daemon', True) for p in self.sub_envs]  # đặt trước khi bắt đầu process
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tuple[TEN, Dict[str,Any]]:
        """
        Reset tất cả môi trường con
        
        :return: Trạng thái ban đầu và thông tin bổ sung
        """
        th.set_grad_enabled(False)

        for pipe in self.sub_pipe1s:
            pipe.send(None)
        states, = self.get_orderly_zip_list_return()
        states = th.tensor(np.stack(states), dtype=th.float32, device=self.device)
        info_dicts = dict()
        return states, info_dicts

    def step(self, action: TEN) -> Tuple[TEN, TEN, TEN, TEN, Dict]:
        """
        Thực hiện một bước trong mỗi môi trường con
        
        :param action: Tensor hành động cho mỗi môi trường con
        :return: Trạng thái mới, phần thưởng, kết thúc, cắt ngắn, và thông tin bổ sung
        """
        action = action.detach().cpu().numpy()
        for pipe, a in zip(self.sub_pipe1s, action):
            pipe.send(a)

        states, rewards, terminal, truncate = self.get_orderly_zip_list_return()
        states = th.tensor(np.stack(states), dtype=th.float32, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        terminal = th.tensor(terminal, dtype=th.bool, device=self.device)
        truncate = th.tensor(truncate, dtype=th.bool, device=self.device)
        info_dicts = dict()
        return states, rewards, terminal, truncate, info_dicts

    def close(self):
        """Đóng tất cả môi trường con"""
        [process.terminate() for process in self.sub_envs]

    def get_orderly_zip_list_return(self):
        """Lấy kết quả từ tất cả môi trường con theo thứ tự"""
        for _ in range(self.num_envs):
            res = self.vec_pipe0.recv()
            self.res_list[res[0]] = res[1:]
        return list(zip(*self.res_list))


