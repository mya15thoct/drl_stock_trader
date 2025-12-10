import os
import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Union, Optional, Tuple, List

from train.config import Config
from train.replay_buffer import ReplayBuffer

TEN = th.Tensor


class AgentBase:
    """
    Lớp cơ sở cho tất cả các Deep Reinforcement Learning agents
    
    :param net_dims: Kích thước của các hidden layers trong mạng neural
    :param state_dim: Số chiều của không gian trạng thái
    :param action_dim: Số chiều của không gian hành động
    :param gpu_id: ID của GPU sử dụng (-1 nếu dùng CPU)
    :param args: Các tham số cấu hình khác
    """
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_discrete: bool = args.if_discrete
        self.if_off_policy: bool = args.if_off_policy

        self.net_dims = net_dims  # Kích thước của các lớp ẩn
        self.state_dim = state_dim  # Số chiều của trạng thái
        self.action_dim = action_dim  # Số chiều của hành động

        self.gamma = args.gamma  # Hệ số chiết khấu
        self.max_step = args.max_step  # Số bước tối đa cho một episode
        self.num_envs = args.num_envs  # Số lượng môi trường vector hóa
        self.batch_size = args.batch_size  # Số lượng mẫu trong một batch
        self.repeat_times = args.repeat_times  # Số lần lặp lại cập nhật
        self.reward_scale = args.reward_scale  # Hệ số scale phần thưởng
        self.learning_rate = args.learning_rate  # Tốc độ học
        self.if_off_policy = args.if_off_policy  # Thuật toán off-policy hoặc on-policy
        self.clip_grad_norm = args.clip_grad_norm  # Cắt gradient để tránh bùng nổ
        self.soft_update_tau = args.soft_update_tau  # Hệ số cập nhật mềm
        self.state_value_tau = args.state_value_tau  # Hệ số chuẩn hóa
        self.buffer_init_size = args.buffer_init_size  # Kích thước tối thiểu buffer trước khi huấn luyện

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # Độ lệch chuẩn của nhiễu khám phá
        self.last_state: Optional[TEN] = None  # Trạng thái cuối cùng trong quá trình
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''Mạng neural'''
        self.act = None  # Mạng Actor
        self.cri = None  # Mạng Critic
        self.act_target = self.act  # Mạng Actor mục tiêu
        self.cri_target = self.cri  # Mạng Critic mục tiêu

        '''Tối ưu hóa'''
        self.act_optimizer: Optional[th.optim] = None
        self.cri_optimizer: Optional[th.optim] = None

        self.criterion = getattr(args, 'criterion', th.nn.MSELoss(reduction="none"))
        self.if_vec_env = self.num_envs > 1  # Sử dụng môi trường vector hóa
        self.if_use_per = getattr(args, 'if_use_per', None)  # Sử dụng Prioritized Experience Replay
        self.lambda_fit_cum_r = getattr(args, 'lambda_fit_cum_r', 0.0)  # Hệ số khớp phần thưởng tích lũy

        """Lưu và tải mô hình"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    def explore_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN]:
        """
        Khám phá môi trường để thu thập dữ liệu huấn luyện
        
        :param env: Môi trường RL
        :param horizon_len: Số bước khám phá
        :return: (states, actions, rewards, undones, unmasks) - Dữ liệu đã thu thập
        """
        if self.if_vec_env:
            return self._explore_vec_env(env=env, horizon_len=horizon_len)
        else:
            return self._explore_one_env(env=env, horizon_len=horizon_len)

    def explore_action(self, state: TEN) -> TEN:
        """
        Chọn hành động để khám phá với nhiễu
        
        :param state: Trạng thái hiện tại
        :return: Hành động với nhiễu
        """
        return self.act.get_action(state, action_std=self.explore_noise_std)

    def _explore_one_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN]:
        """
        Thu thập quỹ đạo thông qua tương tác actor-environment cho một môi trường đơn lẻ
        
        :param env: Môi trường RL
        :param horizon_len: Số bước khám phá
        :return: Tuple (states, actions, rewards, undones, unmasks)
        """
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros(horizon_len, dtype=th.int32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        state = self.last_state
        for t in range(horizon_len):
            action = self.explore_action(state)[0]
            states[t] = state
            actions[t] = action

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device).unsqueeze(0)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state  # Lưu trạng thái cuối cùng
        '''Thêm dim1=1 cho việc nối buffer'''
        states = states.view((horizon_len, 1, self.state_dim))
        actions = actions.view((horizon_len, 1, self.action_dim if not self.if_discrete else 1))
        actions = actions.view((horizon_len, 1, self.action_dim)) \
            if not self.if_discrete else actions.view((horizon_len, 1))
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1))
        unmasks = th.logical_not(truncates).view((horizon_len, 1))
        return states, actions, rewards, undones, unmasks

    def _explore_vec_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN]:
        """
        Thu thập quỹ đạo thông qua tương tác actor-environment cho môi trường vector hóa
        
        :param env: Môi trường RL vector hóa
        :param horizon_len: Số bước khám phá
        :return: Tuple (states, actions, rewards, undones, unmasks)
        """
        states = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, self.num_envs), dtype=th.int32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = self.last_state  # Trạng thái cuối cùng từ lần khám phá trước
        for t in range(horizon_len):
            action = self.explore_action(state)
            states[t] = state
            actions[t] = action

            state, reward, terminal, truncate, _ = env.step(action)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state
        rewards *= self.reward_scale
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        return states, actions, rewards, undones, unmasks

    def update_net(self, buffer: Union[ReplayBuffer, tuple]) -> tuple[float, ...]:
        """
        Cập nhật mạng neural bằng cách lấy mẫu từ buffer
        
        :param buffer: Buffer chứa dữ liệu huấn luyện
        :return: (obj_critic, obj_actor) - Giá trị objective trung bình
        """
        objs_critic = []
        objs_actor = []

        if self.lambda_fit_cum_r != 0:
            buffer.update_cum_rewards(get_cumulative_rewards=self.get_cumulative_rewards)

        th.set_grad_enabled(True)
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer=buffer, update_t=update_t)
            # print(f"DEBUG agent_base.update_net: Received obj_actor: {obj_actor:.4f}")

            objs_critic.append(obj_critic)
            objs_actor.append(obj_actor) if isinstance(obj_actor, float) else None
        th.set_grad_enabled(False)

        obj_avg_critic = np.nanmean(objs_critic) if len(objs_critic) else 0.0
        obj_avg_actor = np.nanmean(objs_actor) if len(objs_actor) else 0.0
        return obj_avg_critic, obj_avg_actor

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> tuple[float, float]:
        """
        Cập nhật mục tiêu cho mạng Actor và Critic
        
        :param buffer: Buffer chứa dữ liệu huấn luyện
        :param update_t: Bước cập nhật hiện tại
        :return: (obj_critic, obj_actor) - Giá trị objective
        """
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_action = self.act(next_state)  # Hành động từ mạng Actor
            next_q = self.cri_target(next_state, next_action)  # Giá trị Q từ mạng Critic mục tiêu

            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action) * unmask
        td_error = self.criterion(q_value, q_label) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
            
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        if_update_act = bool(buffer.cur_size >= self.buffer_init_size)
        if if_update_act:
            action_pg = self.act(state)  # Hành động để tính policy gradient
            obj_actor = self.cri(state, action_pg).mean()
            self.optimizer_backward(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        else:
            obj_actor = th.tensor(th.nan)
        return obj_critic.item(), obj_actor.item()

    def get_cumulative_rewards(self, rewards: TEN, undones: TEN) -> TEN:
        """
        Tính toán phần thưởng tích lũy (phần thưởng có chiết khấu theo thời gian)
        
        :param rewards: Mảng phần thưởng
        :param undones: Mảng trạng thái chưa kết thúc
        :return: Mảng phần thưởng tích lũy
        """
        cum_rewards = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            cum_rewards[t] = next_value = rewards[t] + masks[t] * next_value
        return cum_rewards

    def optimizer_backward(self, optimizer: th.optim, objective: TEN):
        """
        Tối thiểu hóa mục tiêu tối ưu bằng cách cập nhật các tham số mạng
        
        :param optimizer: Tối ưu hóa cho mạng
        :param objective: Mục tiêu tối ưu (thường là hàm mất mát)
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        """
        Cập nhật mềm mạng mục tiêu thông qua mạng hiện tại
        
        :param target_net: Mạng mục tiêu
        :param current_net: Mạng hiện tại
        :param tau: Hệ số cập nhật mềm
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """
        Lưu hoặc tải các file huấn luyện cho Agent
        
        :param cwd: Thư mục làm việc hiện tại
        :param if_save: True để lưu, False để tải
        """
        assert self.save_attr_names.issuperset({'act', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"

            if getattr(self, attr_name) is None:
                continue

            if if_save:
                th.save(getattr(self, attr_name).state_dict(), file_path)
            elif os.path.isfile(file_path):
                getattr(self, attr_name).load_state_dict(th.load(file_path, map_location=self.device))


class ActorBase(nn.Module):
    """
    Lớp cơ sở cho mạng Actor (Policy Network)
    
    :param state_dim: Số chiều của không gian trạng thái
    :param action_dim: Số chiều của không gian hành động
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = None

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_noise_std = None  # Độ lệch chuẩn của nhiễu khám phá
        self.ActionDist = th.distributions.normal.Normal  # Phân phối hành động

    def forward(self, state: TEN) -> TEN:
        """
        Tính toán hành động từ trạng thái
        
        :param state: Trạng thái đầu vào
        :return: Hành động dự đoán trong khoảng [-1, 1]
        """
        action = self.net(state)
        return action.tanh()


class CriticBase(nn.Module):
    """
    Lớp cơ sở cho mạng Critic (Value Network)
    
    :param state_dim: Số chiều của không gian trạng thái
    :param action_dim: Số chiều của không gian hành động
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None

    def forward(self, state: TEN, action: TEN) -> TEN:
        """
        Đánh giá giá trị (Q-value) của cặp (state, action)
        
        :param state: Trạng thái đầu vào
        :param action: Hành động đầu vào
        :return: Giá trị Q
        """
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Giá trị Q

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        """
        Lấy giá trị Q cho cặp (state, action)
        
        :param state: Trạng thái đầu vào
        :param action: Hành động đầu vào
        :return: Giá trị Q
        """
        values = self.net(th.cat((state, action), dim=1))
        return values


"""các hàm tiện ích"""


def build_mlp(dims: List[int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    Xây dựng MLP (MultiLayer Perceptron)
    
    :param dims: Kích thước các lớp, dims[-1] là kích thước đầu ra
    :param activation: Hàm kích hoạt
    :param if_raw_out: True để không áp dụng hàm kích hoạt ở lớp cuối
    :return: Mạng MLP (Sequential)
    """
    if activation is None:
        activation = nn.GELU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # Xóa hàm kích hoạt ở lớp đầu ra để giữ giá trị gốc
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    """
    Khởi tạo các lớp với phương pháp trực giao
    
    :param layer: Lớp cần khởi tạo
    :param std: Độ lệch chuẩn
    :param bias_const: Hằng số cho bias
    """
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class NnReshape(nn.Module):
    """
    Lớp để thay đổi kích thước tensor
    """
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)