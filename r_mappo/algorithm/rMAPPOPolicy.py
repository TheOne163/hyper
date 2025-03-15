import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.utils.util import get_shape_from_obs_space
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

class HyperNetwork(nn.Module):
    def __init__(self, args, obs_space, action_space, device):
        super(HyperNetwork, self).__init__()
        self.args = args
        self.device = device
        self.obs_space = obs_space
        self.action_space = action_space

        # 输入编码器：对每个智能体的 obs 进行编码
        self.obs_encoder = MLPBase(args, obs_space).to(device)

        # 注意力机制：聚合全局信息
        self.attention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=4).to(device)

        # 历史信息处理：LSTM
        self.history_lstm = RNNLayer(
            input_size=args.hidden_size, 
            hidden_size=args.hidden_size, 
            num_layers=1, 
            use_orthogonal=args.use_orthogonal
        ).to(device)

        # Actor 网络参数维度
        self.actor_input_size = get_shape_from_obs_space(obs_space)[0]
        self.actor_hidden_size = 64  # Actor 隐藏层大小
        self.actor_output_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # 输出层：生成 Actor 权重的增量
        self.delta_w1 = nn.Linear(args.hidden_size, self.actor_input_size * self.actor_hidden_size).to(device)
        self.delta_b1 = nn.Linear(args.hidden_size, self.actor_hidden_size).to(device)
        self.delta_w2 = nn.Linear(args.hidden_size, self.actor_hidden_size * self.actor_output_size).to(device)
        self.delta_b2 = nn.Linear(args.hidden_size, self.actor_output_size).to(device)

    def forward(self, obs, rnn_states, history_obs=None, history_actions=None, history_rewards=None):
        """
        输入：
            obs: (batch_size, num_agents, obs_dim)
            rnn_states: (batch_size, num_agents, hidden_size)
            history_obs: (batch_size, seq_len, num_agents, obs_dim) [可选]
        输出：
            delta_w1, delta_b1, delta_w2, delta_b2: Actor 权重增量
            rnn_states: 更新后的 LSTM 状态
        """
        batch_size, num_agents, obs_dim = obs.shape

        # 1. 编码当前 obs
        obs_encoded = self.obs_encoder(obs.view(-1, obs_dim)).view(batch_size, num_agents, -1)

        # 2. 注意力机制聚合全局信息
        obs_encoded = obs_encoded.permute(1, 0, 2)  # (num_agents, batch_size, hidden_size)
        attn_output, _ = self.attention(obs_encoded, obs_encoded, obs_encoded)
        global_info = attn_output.permute(1, 0, 2).mean(dim=1)  # (batch_size, hidden_size)

        # 3. 处理历史信息（可选）
        if history_obs is not None:
            history_input = history_obs.view(batch_size, -1, num_agents * obs_dim)
            history_encoded, rnn_states = self.history_lstm(history_input, rnn_states)
            hyper_input = torch.cat([global_info, history_encoded[:, -1, :]], dim=1)
        else:
            hyper_input = global_info

        # 4. 生成 Actor 权重增量
        delta_w1 = self.delta_w1(hyper_input).view(batch_size, self.actor_input_size, self.actor_hidden_size)
        delta_b1 = self.delta_b1(hyper_input)
        delta_w2 = self.delta_w2(hyper_input).view(batch_size, self.actor_hidden_size, self.actor_output_size)
        delta_b2 = self.delta_b2(hyper_input)

        # 检查编码后的观测是否非零
        print(f"[HyperNet] obs_encoded mean: {obs_encoded.mean().item()}")
        
        # 检查注意力输出
        print(f"[HyperNet] attn_output mean: {attn_output.mean().item()}")
        
        # 检查生成的权重增量
        print(f"[HyperNet] delta_w1 mean: {delta_w1.mean().item()}")
        print(f"[HyperNet] delta_b1 mean: {delta_b1.mean().item()}")

        return delta_w1, delta_b1, delta_w2, delta_b2, rnn_states

#这个类的权重将由超网络生成，而不是直接作为固定的参数存储在模型中
class DynamicActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(DynamicActor, self).__init__()
        self.hidden_size = args.hidden_size
        self.device = device

        # 获取观测和动作空间的维度
        obs_shape = get_shape_from_obs_space(obs_space)
        self.input_size = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
        self.output_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # 初始化基础权重（作为 nn.Parameter）
        self.w1 = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.b1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.w2 = nn.Parameter(torch.zeros(self.hidden_size, self.output_size))
        self.b2 = nn.Parameter(torch.zeros(self.output_size))

        # 如果是连续动作空间，添加 log_std
        if not hasattr(action_space, 'n'):
            self.log_std = nn.Parameter(torch.zeros(self.output_size))

        self.to(device)

    def forward(self, obs, delta_w1, delta_b1, delta_w2, delta_b2):
        """
        前向传播，动态更新权重并生成动作分布。
        :param obs: (batch_size, obs_dim) 观测输入
        :param delta_w1: (batch_size, input_size, hidden_size) 权重增量
        :param delta_b1: (batch_size, hidden_size) 偏置增量
        :param delta_w2: (batch_size, hidden_size, output_size) 权重增量
        :param delta_b2: (batch_size, output_size) 偏置增量
        :return: 动作分布
        """
        # 动态更新权重
        w1 = self.w1 + delta_w1
        b1 = self.b1 + delta_b1
        w2 = self.w2 + delta_w2
        b2 = self.b2 + delta_b2

        # 前向传播
        hidden = torch.relu(torch.matmul(obs, w1) + b1)
        output = torch.matmul(hidden, w2) + b2

        # 根据动作空间返回分布
        if hasattr(self, 'log_std'):  # 连续动作空间
            std = torch.exp(self.log_std)
            return Normal(output, std)
        else:  # 离散动作空间
            return Categorical(logits=output)


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self._use_smooth_regularizer = args.use_smooth_regularizer  #添加超网络和修改初始化
        self._use_align_regularizer = args.use_align_regularizer    #添加超网络和修改初始化

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.hyper_net = HyperNetwork(args, obs_space, act_space, device)   #添加超网络和修改初始化
        self.actor = DynamicActor(args, obs_space, act_space, device).to(device)   #修改初始化


    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
        #                                                              rnn_states_actor,
        #                                                              action,
        #                                                              masks,
        #                                                              available_actions,
        #                                                              active_masks)

        # values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        # return values, action_log_probs, dist_entropy
        
        # 生成权重增量
        delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(obs, rnn_states_actor)

        action_log_probs = []
        dist_entropy = []
        dists = []  # 保存动作分布
        for i in range(num_agents):
            agent_obs = obs[:, i, :]
            dist = self.actor(agent_obs, delta_w1, delta_b1, delta_w2, delta_b2)
            action_log_probs.append(dist.log_prob(action[:, i, :]))
            dist_entropy.append(dist.entropy().mean())
            dists.append(dist)

        action_log_probs = torch.stack(action_log_probs, dim=1)
        dist_entropy = torch.stack(dist_entropy, dim=1).mean()
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, delta_w1, delta_b1, delta_w2, delta_b2, dists

    # def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
    #     """
    #     Compute actions using the given inputs.
    #     :param obs (np.ndarray): local agent inputs to the actor.
    #     :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
    #     :param masks: (np.ndarray) denotes points at which RNN states should be reset.
    #     :param available_actions: (np.ndarray) denotes which actions are available to agent
    #                               (if None, all actions available)
    #     :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
    #     """
    #     actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
    #     return actions, rnn_states_actor
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
    """
    输入：
        obs: (batch_size, num_agents, obs_dim)
        rnn_states_actor: (batch_size, num_agents, hidden_size)
    输出：
        actions, action_log_probs, rnn_states_actor
    """
    batch_size, num_agents, obs_dim = obs.shape

    # 生成权重增量
    delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(obs, rnn_states_actor)

     # 检查权重增量是否非零
    print(f"[Debug] delta_w1 mean: {delta_w1.mean().item()}, delta_b1 mean: {delta_b1.mean().item()}")

    actions = []
    action_log_probs = []
    for i in range(obs.shape[1]):   # num_agents
        agent_obs = obs[:, i, :]
        dist = self.actor(agent_obs, delta_w1, delta_b1, delta_w2, delta_b2)
        dist = self.actor(agent_obs, agent_delta_w1, agent_delta_b1, agent_delta_w2, agent_delta_b2)
        action = dist.mode() if deterministic else dist.sample()
        actions.append(action)
        action_log_probs.append(dist.log_prob(action))

    actions = torch.stack(actions, dim=1)
    action_log_probs = torch.stack(action_log_probs, dim=1)

    return actions, action_log_probs, rnn_states_actor