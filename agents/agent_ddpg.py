import numpy as np
import torch as th
from copy import deepcopy
from typing import Tuple, List, Union

from .agent_base import AgentBase, ActorBase, CriticBase, build_mlp, layer_init_with_orthogonal
from train.replay_buffer import ReplayBuffer
from train.config import Config

TEN = th.Tensor


class AgentDDPG(AgentBase):
    """
    DDPG (Deep Deterministic Policy Gradient) - Actor-Critic algorithm cho không gian hành động liên tục
    

    :param net_dims: Kích thước của các hidden layers trong mạng neural
    :param state_dim: Số chiều của không gian trạng thái
    :param action_dim: Số chiều của không gian hành động (1 cho cổ phiếu FPT)
    :param gpu_id: ID của GPU sử dụng (-1 nếu dùng CPU)
    :param args: Các tham số cấu hình khác
    """
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.1)  # Độ lệch chuẩn của nhiễu khám phá
        
        # Khởi tạo mạng Actor và Critic
        self.act = Actor(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = Critic(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        
        # Khởi tạo mạng target (sao chép từ mạng chính)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        
        # Khởi tạo bộ tối ưu hóa
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
    
    def update_objectives(self, buffer: Union[ReplayBuffer, tuple], update_t: int) -> Tuple[float, float]:
        """
        Cập nhật các mạng Actor và Critic
        
        :param buffer: Buffer chứa các trải nghiệm (state, action, reward, undone)
        :param update_t: Bước cập nhật hiện tại
        :return: (obj_critic, obj_actor) - Giá trị objective của Critic và Actor
        """
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None
            
            # Tính giá trị Q cho trạng thái tiếp theo sử dụng mạng target
            next_action = self.act_target(next_state)  # Chính sách xác định (deterministic policy)
            next_q = self.cri_target(next_state, next_action)
            
            # Tính giá trị Q mục tiêu (target Q-value)
            q_label = reward + undone * self.gamma * next_q
        
        # Cập nhật mạng Critic (đánh giá giá trị hành động)
        q_value = self.cri(state, action) * unmask
        td_error = self.criterion(q_value, q_label) * unmask
        
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        
        # Cập nhật mạng Critic (chức năng đánh giá giá trị)
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        
        # Cập nhật mạng Actor (chính sách)
        if buffer.cur_size >= self.buffer_init_size:  # Chỉ cập nhật Actor sau khi đã thu thập đủ kinh nghiệm
            action_pg = self.act(state)  # Action từ mạng Actor hiện tại
            obj_actor = -self.cri(state, action_pg).mean()  # Maximize Q-value
            
            self.optimizer_backward(self.act_optimizer, obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
            
            return obj_critic.item(), obj_actor.item()
        else:
            return obj_critic.item(), float('nan')
    
    def explore_action(self, state: TEN) -> TEN:
        """
        Khám phá hành động với nhiễu Gaussian
        
        :param state: Trạng thái hiện tại
        :return: Hành động với nhiễu khám phá
        """
        action = self.act(state)
        noise = th.randn_like(action) * self.explore_noise_std
        return (action + noise).clamp(-1.0, 1.0)  # Clip hành động vào khoảng [-1, 1]
    
    def get_action(self, state: TEN, if_deterministic=False) -> np.ndarray:
        """
        Lấy hành động từ actor network
        
        :param state: Trạng thái hiện tại
        :param if_deterministic: Nếu True, không thêm nhiễu khám phá
        :return: Hành động được chọn
        """
        state_tensor = th.as_tensor(state, dtype=th.float32, device=self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        action_tensor = self.act(state_tensor)
        if not if_deterministic:
            noise = th.randn_like(action_tensor) * self.explore_noise_std
            action_tensor = (action_tensor + noise).clamp(-1.0, 1.0)
            
        return action_tensor.detach().cpu().numpy()


class Actor(ActorBase):
    """
    Mạng Actor trong DDPG
    """
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)
    
    def forward(self, state: TEN) -> TEN:
        """
        Tính toán hành động từ trạng thái
        
        :param state: Trạng thái đầu vào
        :return: Hành động dự đoán trong khoảng [-1, 1]
        """
        return self.net(state).tanh()
    
    def get_action(self, state: TEN, action_std: float = 0.0) -> TEN:
        """
        Lấy hành động từ trạng thái, có thể thêm nhiễu Gaussian
        
        :param state: Trạng thái đầu vào
        :param action_std: Độ lệch chuẩn của nhiễu (0 nếu không thêm nhiễu)
        :return: Hành động với nhiễu (nếu có) trong khoảng [-1, 1]
        """
        action = self.forward(state)
        if action_std > 0:
            noise = th.randn_like(action) * action_std
            action = (action + noise).clamp(-1.0, 1.0)
        return action


class Critic(CriticBase):
    """
    Mạng Critic trong DDPG
    """
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)
    
    def forward(self, state: TEN, action: TEN) -> TEN:
        """
        Đánh giá giá trị (Q-value) của cặp (state, action)
        
        :param state: Trạng thái đầu vào
        :param action: Hành động đầu vào
        :return: Giá trị Q
        """
        input_tensor = th.cat((state, action), dim=1)
        return self.net(input_tensor)