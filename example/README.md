# SandboxFusion 使用示例

本目录包含 SandboxFusion 的使用示例，特别针对 **RL 训练 Coding Model** 的场景。

## 环境准备

```bash
# 1. 激活 conda 环境
conda activate sandbox

# 2. 启动 SandboxFusion 服务
cd /Users/xiaohui/Desktop/verl/SandboxFusion
make run

# 3. 运行示例（新终端）
conda activate sandbox
python example/01_basic_run_code.py
```

## 示例文件

### 1. 基础代码执行 (`01_basic_run_code.py`)

展示如何执行 Python、C++、Go 等语言的代码：
- 基础代码执行
- 带标准输入的执行
- 超时处理

```python
from sandbox_fusion import run_code, RunCodeRequest

response = run_code(RunCodeRequest(
    code='print("Hello, World!")',
    language='python'
))
print(response.run_result.stdout)
```

### 2. 多语言支持 (`02_multi_language.py`)

展示 30+ 种编程语言的执行：
- Java, Node.js, TypeScript
- Rust, Ruby, Bash 等

### 3. RL 训练数据集使用 (`03_dataset_for_rl_training.py`) ⭐

**重点示例**：展示如何在 RL 训练中使用数据集

```python
from sandbox_fusion import get_prompts, submit, GetPromptsRequest, SubmitRequest, TestConfig

# 1. 获取数据集 prompts（作为模型输入）
prompts = get_prompts(GetPromptsRequest(
    dataset='humaneval',
    config={'language': 'python'}
))

# 2. 模型生成代码
completion = your_model.generate(prompts[0].prompt)

# 3. 提交评测获取 reward
result = submit(SubmitRequest(
    dataset='humaneval',
    id=prompts[0].id,
    completion=completion,
    config=TestConfig(language='python')
))

# 4. 计算 reward
reward = 1.0 if result.accepted else 0.0
```

包含内容：
- 获取 HumanEval、MBPP、PAL-Math 等数据集
- 提交代码评测
- 批量并发评测
- 自定义数据评测
- RL 训练循环示例

### 4. RL Reward 集成 (`04_rl_reward_integration.py`) ⭐

**进阶示例**：完整的 RL 训练集成

```python
from example.rl_reward_integration import RewardCalculator, DatasetLoader

# 初始化
calculator = RewardCalculator(
    dataset='humaneval',
    pass_reward=1.0,
    fail_reward=0.0,
    partial_reward=True  # 部分通过也给分
)

# 批量评测
results = calculator.evaluate_batch(samples, max_workers=10)
rewards = [r.reward for r in results]

# 用于更新模型
# loss = compute_policy_gradient_loss(log_probs, rewards)
```

包含内容：
- `RewardCalculator`: 封装 reward 计算逻辑
- `DatasetLoader`: 数据集加载和采样
- 批量评测和并发处理
- veRL 框架集成示例
- 异步评测示例

## 支持的数据集

| 数据集 | 说明 | 题目数 | 适用场景 |
|--------|------|--------|----------|
| `humaneval` | OpenAI 代码生成基准 | 164 | 代码生成能力评估 |
| `mbpp` | Google 基础编程问题 | 500 | 基础编程能力 |
| `palmath` | 数学推理问题 | 多个子集 | 数学推理能力 |
| `codecontests` | 竞赛编程问题 | 大量 | 算法能力 |
| `cruxeval` | 代码理解问题 | - | 代码理解 |
| `naturalcodebench` | 自然语言编程 | 402 | 真实场景 |

## RL 训练集成指南

### 典型训练流程

```python
# 1. 初始化
calculator = RewardCalculator(dataset='humaneval')
loader = DatasetLoader(dataset='humaneval')

# 2. 训练循环
for epoch in range(num_epochs):
    # 采样问题
    batch = loader.sample(batch_size)

    # 模型生成代码
    completions = model.generate([b['prompt'] for b in batch])

    # 获取 rewards
    samples = [CodeSample(b['id'], b['prompt'], c) for b, c in zip(batch, completions)]
    results = calculator.evaluate_batch(samples)
    rewards = [r.reward for r in results]

    # 更新模型
    loss = compute_loss(log_probs, rewards)
    optimizer.step()
```

### veRL 集成

```python
def code_reward_fn(prompts, completions, problem_ids, **kwargs):
    calculator = RewardCalculator(dataset=kwargs['dataset'])
    samples = [CodeSample(pid, p, c) for pid, p, c in zip(problem_ids, prompts, completions)]
    results = calculator.evaluate_batch(samples)
    return [r.reward for r in results]

# veRL 配置
config.reward_fn = code_reward_fn
```

## API 快速参考

### 获取数据集

```python
# 获取所有问题
prompts = get_prompts(GetPromptsRequest(dataset='humaneval', config={'language': 'python'}))

# 获取单个问题
prompt = get_prompt_by_id(GetPromptByIdRequest(dataset='humaneval', id='0'))
```

### 提交评测

```python
# 普通提交（失败会抛异常）
result = submit(SubmitRequest(dataset='humaneval', id='0', completion=code, config=TestConfig()))

# 安全提交（失败返回结果，不抛异常）
result = submit_safe(...)

# 异步提交
result = await submit_async(...)
```

### 批量评测

```python
# 并发评测
results = run_concurrent(func=submit_safe, args_list=requests, max_workers=10)
```

## 常见问题

### 1. 连接失败

确保 SandboxFusion 服务正在运行：
```bash
make run
```

### 2. 超时

增加超时时间：
```python
config=TestConfig(run_timeout=60)
```

### 3. 内存限制

设置内存限制：
```python
RunCodeRequest(memory_limit_MB=1024)
```
