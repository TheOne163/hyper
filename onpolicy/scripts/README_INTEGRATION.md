# 超网络与正则化集成指南

## 问题诊断

在集成超网络和正则化过程中，我们遇到并解决了以下问题：

### 2024-03-16 更新

1. **HyperNetwork 初始化问题**
   - 修复了 `RNNLayer` 的初始化参数
   - 将 `input_size` 改为 `inputs_dim`
   - 将 `hidden_size` 改为 `outputs_dim`
   - 将 `num_layers` 改为 `recurrent_N`

2. **输入维度处理**
   - 添加了对 NumPy 数组输入的处理
   - 自动转换为 PyTorch 张量并移动到正确的设备
   - 处理 2D 和 3D 输入张量

3. **动作生成问题**
   - 修复了动作超出范围的问题
   - 使用 `torch.bmm` 进行批量矩阵乘法
   - 正确处理动作维度和类型

4. **概率分布处理**
   - 使用 `softmax` 确保概率分布合法
   - 创建新的 `Categorical` 分布而不是修改现有分布
   - 正确处理可用动作掩码

### 主要修改文件

1. **rMAPPOPolicy.py**
   - `HyperNetwork` 类：修复了 RNN 初始化和输入处理
   - `DynamicActor` 类：改进了动作生成和概率分布处理
   - `act` 方法：正确处理可用动作掩码

## 接口说明

### HyperNetwork

```python
def forward(self, obs, rnn_states, history_obs=None):
    """
    输入：
        obs: (batch_size, obs_dim) 或 (batch_size, num_agents, obs_dim)
        rnn_states: (batch_size, hidden_size)
        history_obs: (batch_size, seq_len, num_agents, obs_dim) [可选]
    输出：
        delta_w1, delta_b1, delta_w2, delta_b2: Actor 权重增量
        rnn_states: 更新后的 LSTM 状态
    """
```

### DynamicActor

```python
def forward(self, obs, delta_w1, delta_b1, delta_w2, delta_b2):
    """
    输入：
        obs: (batch_size, obs_dim)
        delta_w1: (batch_size, input_size, hidden_size)
        delta_b1: (batch_size, hidden_size)
        delta_w2: (batch_size, hidden_size, output_size)
        delta_b2: (batch_size, output_size)
    输出：
        动作分布（Categorical 或 Normal）
    """
```

## 注意事项

1. 确保输入数据类型正确（PyTorch 张量或 NumPy 数组）
2. 注意维度匹配，特别是在处理批量数据时
3. 使用可用动作掩码时，确保概率分布正确归一化
4. 检查生成的动作是否在有效范围内

## 调试信息

代码中添加了以下调试信息输出：
- 超网络生成的权重增量均值
- 编码后的观测均值
- 注意力输出均值
- 动作分布的概率值

## 1. 问题诊断

### 1.1 模块调用链路
当前训练流程：
```runner.run() -> collect() -> compute() -> train() -> trainer.ppo_update() -> policy.evaluate_actions()
```

### 1.2 发现的问题
1. 超网络模块未被正确调用
2. 正则化损失未被计算和使用
3. 参数传递不完整

## 2. 需要修改的文件

### 2.1 rMAPPOPolicy.py
```python
# 1. 修改初始化
def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
    # ... 现有代码 ...
    self.hyper_net = HyperNetwork(args, self.obs_space, self.act_space, self.device)
    self.actor = DynamicActor(args, self.obs_space, self.act_space, self.device)

# 2. 修改evaluate_actions方法
def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks, available_actions=None, active_masks=None):
    # 生成权重增量
    delta_w1, delta_b1, delta_w2, delta_b2, rnn_states = self.hyper_net(obs, rnn_states_actor)
    
    # 使用动态Actor评估动作
    values, action_log_probs, dist_entropy = self.actor(
        obs, delta_w1, delta_b1, delta_w2, delta_b2
    )
    
    return values, action_log_probs, dist_entropy, delta_w1, delta_b1, delta_w2, delta_b2

# 3. 修改get_actions方法
def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
    # 类似的修改...
```

### 2.2 r_mappo.py (trainer)
```python
def ppo_update(self, sample, update_actor=True):
    # ... 现有代码 ...
    
    # 获取评估结果，包括权重增量
    values, action_log_probs, dist_entropy, delta_w1, delta_b1, delta_w2, delta_b2 = \
        self.policy.evaluate_actions(...)
    
    # 计算正则化损失
    if self.policy.use_smooth_regularizer:
        l_smooth = self.compute_smooth_regularizer(delta_w1, delta_b1, delta_w2, delta_b2)
        policy_loss += self.policy.smooth_weight * l_smooth
        
    if self.policy.use_align_regularizer:
        l_align = self.compute_align_regularizer(delta_w1, delta_b1, delta_w2, delta_b2)
        policy_loss += self.policy.align_weight * l_align
```

## 3. 调试验证点

### 3.1 超网络调用验证
在 `rMAPPOPolicy.py` 中添加调试信息：
```python
def evaluate_actions(self, ...):
    print("[Debug] HyperNetwork forward called")
    print(f"[Debug] Generated weight deltas: {delta_w1.mean():.4f}, {delta_b1.mean():.4f}")
```

### 3.2 正则化损失验证
在 `r_mappo.py` 中添加调试信息：
```python
def ppo_update(self, ...):
    if self.policy.use_smooth_regularizer:
        print(f"[Debug] Smooth loss: {l_smooth.item():.4f}")
    if self.policy.use_align_regularizer:
        print(f"[Debug] Align loss: {l_align.item():.4f}")
```

## 4. 待完成任务

- [ ] 实现 `compute_smooth_regularizer` 方法
- [ ] 实现 `compute_align_regularizer` 方法
- [ ] 添加正则化权重的衰减机制
- [ ] 添加超网络参数的梯度裁剪
- [ ] 完善日志记录系统，添加正则化损失的跟踪

## 5. 注意事项

1. 确保设备一致性（CPU/GPU）
2. 注意梯度累积和反向传播的顺序
3. 监控正则化损失的数值范围
4. 确保超网络的输出维度与动态Actor的输入维度匹配

## 6. 性能优化建议

1. 考虑使用 `torch.jit.script` 优化超网络计算
2. 添加权重增量的范围限制
3. 实现正则化损失的并行计算
4. 添加早停机制，避免过度正则化 