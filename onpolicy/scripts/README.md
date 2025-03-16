# 修改记录

## 超网络（HyperNetwork）相关修改

### 1. 文件结构
- `onpolicy/algorithms/r_mappo/algorithm/hyper_network.py`: 包含 HyperNetwork 和 DynamicActor 类的定义
- `onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`: 主策略文件

### 2. 主要修改内容

#### 2.1 参数添加
在 `train_smac.py` 中添加了以下参数：
```python
parser.add_argument("--use_smooth_regularizer", action='store_true', default=True)
parser.add_argument("--use_align_regularizer", action='store_true', default=True)
parser.add_argument("--smooth_weight", type=float, default=0.1)
parser.add_argument("--align_weight", type=float, default=0.05)
```

#### 2.2 rMAPPOPolicy.py 修改
1. 移除了重复的 HyperNetwork 和 DynamicActor 类定义
2. 修改了 `__init__` 方法中的超网络初始化
3. 更新了 `evaluate_actions` 和 `act` 方法以支持超网络

### 3. 调用流程
1. HyperNetwork 生成权重增量（delta_w1, delta_b1, delta_w2, delta_b2）
2. DynamicActor 使用基础权重和增量生成动作分布
3. 根据动作分布采样或选择确定性动作

### 4. 注意事项
- 确保超网络和动态Actor的输入输出维度匹配
- 权重增量的大小需要合理控制
- 注意参数的设备（CPU/GPU）一致性

### 5. 待优化项
- [ ] 添加权重增量的范围限制
- [ ] 优化超网络的注意力机制
- [ ] 添加历史信息的处理
- [ ] 完善正则化项的实现 