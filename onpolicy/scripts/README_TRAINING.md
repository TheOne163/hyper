# MAPPO 训练流程与超网络集成说明

## 1. 训练流程概述

训练过程从 `train.py` 开始，主要包含以下步骤：

### 1.1 初始化阶段
1. 创建环境
2. 初始化策略网络（Policy）
   - 包含超网络（HyperNetwork）
   - 动态Actor（DynamicActor）
   - Critic网络
3. 初始化训练器（R_MAPPO）

### 1.2 训练循环
```
train.py -> runner.run() -> 训练循环:
    for episode in range(episodes):
        for step in range(episode_length):
            1. collect() - 采样动作
            2. env.step() - 环境交互
            3. insert() - 存储数据
        compute() - 计算回报
        train() - 更新网络
```

## 2. 超网络调用链

### 2.1 动作生成过程
1. `runner.collect()` 调用 `policy.get_actions()`
2. `get_actions()` 中的超网络调用：
```python
# 1. 使用超网络生成权重增量
delta_w1, delta_b1, delta_w2, delta_b2, rnn_states_actor = self.hyper_net(
    obs, 
    rnn_states_actor
)

# 2. 使用动态Actor生成动作
dist = self.actor(obs, delta_w1, delta_b1, delta_w2, delta_b2)
```

### 2.2 策略更新过程
1. `R_MAPPO.train()` 调用 `ppo_update()`
2. `ppo_update()` 中的超网络调用：
   - 通过 `evaluate_actions()` 评估动作
   - 计算策略损失和价值损失
   - 更新超网络和动态Actor的参数

## 3. 关键函数说明

### 3.1 HyperNetwork.forward()
```python
输入：
- obs: (batch_size, num_agents, obs_dim)
- rnn_states: (batch_size, num_agents, hidden_size)
- history_obs: (batch_size, seq_len, num_agents, obs_dim) [可选]

处理步骤：
1. 编码观测
2. 使用注意力机制聚合全局信息
3. 处理历史信息（如果有）
4. 生成Actor权重增量

输出：
- delta_w1, delta_b1, delta_w2, delta_b2: Actor权重增量
- rnn_states: 更新后的LSTM状态
```

### 3.2 DynamicActor.forward()
```python
输入：
- obs: (batch_size, obs_dim)
- delta_w1, delta_b1, delta_w2, delta_b2: 权重增量

处理步骤：
1. 更新网络权重
2. 前向传播生成动作
3. 返回动作分布

输出：
- 动作分布（Categorical或Normal）
```

## 4. 超网络集成验证点

1. 初始化检查：
   - 确保超网络和动态Actor正确初始化
   - 验证优化器包含所有相关参数

2. 前向传播检查：
   - 超网络生成的权重增量是否非零
   - 动态Actor使用更新后的权重是否正确

3. 反向传播检查：
   - 超网络参数是否得到更新
   - 梯度是否正确传播

4. 多智能体协调：
   - 确保每个智能体的超网络正确处理观测
   - 验证注意力机制的全局信息聚合

## 5. 调试信息

在代码中添加了以下调试打印：
```python
print(f"[HyperNet] obs_encoded mean: {obs_encoded.mean().item()}")
print(f"[HyperNet] attn_output mean: {attn_output.mean().item()}")
print(f"[HyperNet] delta_w1 mean: {delta_w1.mean().item()}")
print(f"[HyperNet] delta_b1 mean: {delta_b1.mean().item()}")
```

## 6. 注意事项

1. 超网络和动态Actor的参数维度必须匹配
2. 确保超网络生成的权重增量规模适当
3. 注意多智能体场景下的观测处理
4. 监控训练过程中的权重更新幅度 