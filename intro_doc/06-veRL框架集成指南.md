# veRL 框架集成指南

本文档详细介绍如何在 veRL 强化学习框架中使用 SandboxFusion 进行代码评测，为 RL 训练提供 reward 信号。

---

## 概述

veRL 是一个用于训练大语言模型的强化学习框架。通过集成 SandboxFusion，可以在训练过程中对模型生成的代码进行实时评测，将评测结果作为 reward 信号来优化模型的代码生成能力。

### 核心流程

```
模型生成代码 → SandboxFusion 评测 → 计算 Reward → 更新模型
```

---

## 快速开始

### 启动脚本示例

```bash
# verl/examples/ppo_trainer/run_deepseek7b_llm_sandbox_fusion.sh

python3 -m verl.trainer.main_ppo \
    reward_model.sandbox_fusion.url='http://localhost:8080/run_code' \
    reward_model.sandbox_fusion.max_concurrent=128 \
    reward_model.reward_manager=prime \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/validation.parquet \
    actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
    # ... 其他配置
```

### 关键配置参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `reward_model.sandbox_fusion.url` | SandboxFusion API 端点 | `http://localhost:8080/run_code` |
| `reward_model.sandbox_fusion.max_concurrent` | 最大并发请求数 | `128` |
| `reward_model.sandbox_fusion.memory_limit_mb` | 内存限制 (MB) | `1024` |
| `reward_model.reward_manager` | 奖励管理器类型 | `prime` |

---

## 架构详解

### 1. 配置体系

#### SandboxFusionConfig

```python
# verl/workers/config/reward_model.py

@dataclass
class SandboxFusionConfig(BaseConfig):
    """SandboxFusion 配置"""
    url: Optional[str] = None           # API 端点 URL
    max_concurrent: int = 64            # 最大并发数
    memory_limit_mb: int = 1024         # 内存限制 (MB)

@dataclass
class RewardModelConfig(BaseConfig):
    """奖励模型配置"""
    reward_manager: Optional[str] = None  # 奖励管理器名称
    enable: bool = False
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
```

### 2. 奖励管理器

veRL 提供多种奖励管理器，推荐使用 `prime`：

| 管理器 | 说明 | 适用场景 |
|--------|------|----------|
| `prime` | 支持并发评分，使用进程池 | 代码评测（推荐） |
| `naive` | 简单串行评分 | 调试 |
| `batch` | 批量评分 | 大批量处理 |
| `dapo` | DAPO 算法专用 | DAPO 训练 |

#### Prime 奖励管理器

```python
# verl/workers/reward_manager/prime.py

@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.compute_score = compute_score or default_compute_score

    def __call__(self, data: DataProto, return_dict=False):
        # 1. 解码模型生成的序列
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # 2. 并发执行评分
        scores = self.verify(data)

        # 3. 构建 reward tensor
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i] - 1] = scores[i]

        return {"reward_tensor": reward_tensor}

    def verify(self, data):
        """使用进程池并发评分"""
        scores = run_reward_scoring(
            self.compute_score,
            completions=sequences_str,
            references=ground_truth,
            tasks=data_sources,
            num_processes=64,
        )
        return scores
```

### 3. 评分路由

`default_compute_score` 根据数据源路由到不同的评分函数：

```python
# verl/utils/reward_score/__init__.py

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """根据数据源路由评分函数"""

    # 代码评测数据源
    if data_source in ["codecontests", "apps", "codeforces", "taco"]:
        if sandbox_fusion_url:
            from . import sandbox_fusion
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb,
                solution_str, ground_truth, continuous=True
            )
        else:
            from . import prime_code
            res = prime_code.compute_score(solution_str, ground_truth)

    # 数学推理数据源
    elif data_source in ["openai/gsm8k"]:
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ["lighteval/MATH"]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)

    return res
```

**支持的数据源**：
- 代码评测：`codecontests`, `apps`, `codeforces`, `taco`
- 数学推理：`openai/gsm8k`, `lighteval/MATH`
- 自定义：可以扩展支持更多数据源

### 4. SandboxFusion API 调用

```python
# verl/utils/reward_score/sandbox_fusion/__init__.py

def compute_score(
    sandbox_fusion_url, concurrent_semaphore, memory_limit_mb,
    completion, test_cases, continuous=False, timeout=10
):
    """使用 SandboxFusion 评测代码"""

    # 1. 提取代码块
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]

    # 2. 调用 API 执行测试
    res_list, metadata_list = check_correctness(
        sandbox_fusion_url=sandbox_fusion_url,
        in_outs=test_cases,
        generation=solution,
        timeout=timeout,
        concurrent_semaphore=concurrent_semaphore,
        memory_limit_mb=memory_limit_mb,
    )

    # 3. 计算评分（通过率）
    if continuous:
        # 前 N 个测试用例的通过率
        num_to_consider = min(len(res_list), 10)
        passed = sum(1 for r in res_list[:num_to_consider] if r is True)
        score = passed / num_to_consider
    else:
        passed = sum(1 for r in res_list if r is True)
        score = passed / len(res_list) if res_list else 0.0

    return float(score), metadata_list
```

### 5. 并发控制

```python
# verl/utils/reward_score/sandbox_fusion/utils.py

def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
):
    """调用 SandboxFusion API"""

    payload = {
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "code": code,
        "stdin": stdin,
        "memory_limit_MB": memory_limit_mb,
        "language": language,
    }

    # 重试逻辑（处理 504 Gateway Timeout）
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                sandbox_fusion_url,
                json=payload,
                timeout=compile_timeout + run_timeout + API_TIMEOUT,
            )

            if response.status_code == 504:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue

            return response.json(), None
        except Exception as e:
            return None, str(e)


def check_correctness(sandbox_fusion_url, in_outs, generation, timeout,
                      concurrent_semaphore, memory_limit_mb):
    """并发检查代码正确性"""

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(
                _process_single_case,
                i, stdin, expected,
                sandbox_fusion_url, generation,
                timeout, memory_limit_mb,
                concurrent_semaphore,  # 信号量控制并发
            ): i
            for i, (stdin, expected) in enumerate(zip(inputs, outputs))
        }

        for future in as_completed(futures):
            result, metadata = future.result()
            results[futures[future]] = result

    return results, metadata_list
```

---

## 数据流完整链路

```
┌─────────────────────────────────────────────────────────────────┐
│                      veRL 训练循环                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 数据加载                                                     │
│     └── train.parquet (prompts + test_cases)                   │
│                    ↓                                            │
│  2. Actor 生成代码                                               │
│     └── model.generate(prompt) → completion                    │
│                    ↓                                            │
│  3. 计算 Reward                                                  │
│     └── reward_fn(data)                                        │
│              ↓                                                  │
│     ┌────────────────────────────────────────┐                 │
│     │  PrimeRewardManager                    │                 │
│     │    └── verify(data)                    │                 │
│     │         └── run_reward_scoring()       │                 │
│     │              └── default_compute_score()│                │
│     │                   └── sandbox_fusion.compute_score()     │
│     │                        └── check_correctness()           │
│     │                             └── call_sandbox_api() ──────┼──→ SandboxFusion
│     └────────────────────────────────────────┘                 │
│                    ↓                                            │
│  4. 更新模型                                                     │
│     └── PPO loss + reward → optimizer.step()                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 评测结果状态码

| 状态码 | 含义 | 说明 |
|--------|------|------|
| `True` | 测试通过 | 输出与期望一致 |
| `False` | 答案错误 | Wrong Answer |
| `-1` | API 错误 | 沙箱服务异常 |
| `-2` | 运行时错误 | Runtime Error |
| `-3` | 超时 | Time Limit Exceeded |
| `-4` | 编译错误 | Compile Error |

---

## 配置示例

### 完整配置文件

```yaml
# config/ppo_trainer.yaml

reward_model:
  reward_manager: prime
  sandbox_fusion:
    url: "http://localhost:8080/run_code"
    max_concurrent: 128
    memory_limit_mb: 1024

data:
  train_files: ${HOME}/data/train.parquet
  val_files: ${HOME}/data/validation.parquet
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 512

actor_rollout_ref:
  model:
    path: deepseek-ai/deepseek-llm-7b-chat
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 16

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  total_epochs: 15
  save_freq: 20
  test_freq: 1
```

### 数据格式要求

训练数据 (parquet) 应包含以下字段：

```python
{
    "prompt": "实现一个函数...",           # 输入提示
    "data_source": "codecontests",        # 数据源标识
    "ground_truth": {                     # 测试用例
        "inputs": ["1 2\n", "3 4\n"],
        "outputs": ["3\n", "7\n"]
    }
}
```

---

## 自定义扩展

### 添加新的数据源

```python
# verl/utils/reward_score/__init__.py

def default_compute_score(data_source, solution_str, ground_truth,
                          sandbox_fusion_url=None, **kwargs):

    # 添加新的数据源处理
    if data_source == "my_custom_dataset":
        from . import my_custom_reward
        return my_custom_reward.compute_score(solution_str, ground_truth)

    # ... 其他数据源
```

### 自定义奖励管理器

```python
# my_reward_manager.py

from verl.workers.reward_manager.registry import register
from verl.workers.reward_manager.base import AbstractRewardManager

@register("my_custom")
class MyCustomRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, compute_score=None, **kwargs):
        self.tokenizer = tokenizer
        self.compute_score = compute_score

    def __call__(self, data, return_dict=False):
        # 自定义评分逻辑
        sequences = self.tokenizer.batch_decode(data.batch["responses"])

        rewards = []
        for seq in sequences:
            score = self.compute_score(seq, ...)
            rewards.append(score)

        reward_tensor = torch.tensor(rewards)
        return {"reward_tensor": reward_tensor} if return_dict else reward_tensor
```

使用自定义管理器：

```bash
python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=my_custom \
    # ...
```

---

## 部署建议

### 1. SandboxFusion 服务部署

```bash
# 本地部署
cd SandboxFusion
make run  # 默认端口 8080

# Docker 部署（推荐生产环境）
docker run -d --rm --privileged -p 8080:8080 code_sandbox:server
```

### 2. 并发配置建议

| 场景 | max_concurrent | 说明 |
|------|----------------|------|
| 本地开发 | 16-32 | 避免资源竞争 |
| 单机训练 | 64-128 | 充分利用 CPU |
| 分布式训练 | 128-256 | 根据服务端能力调整 |

### 3. 超时配置

```python
# 建议值
compile_timeout = 10   # 编译超时
run_timeout = 10       # 运行超时
memory_limit_mb = 1024 # 内存限制
```

---

## 常见问题

### 1. 504 Gateway Timeout

**原因**: SandboxFusion 服务过载或响应慢

**解决**:
- 降低 `max_concurrent`
- 增加服务端资源
- 检查网络延迟

### 2. Reward 全为 0

**原因**: 代码提取失败或测试用例格式错误

**排查**:
```python
# 检查代码提取
completion = "```python\ndef solution():\n    pass\n```"
if "```python" in completion:
    code = completion.split("```python")[-1].split("```")[0]
    print(code)

# 检查测试用例格式
test_cases = {"inputs": [...], "outputs": [...]}
```

### 3. 内存不足

**解决**:
- 降低 `memory_limit_mb`
- 减少 `max_concurrent`
- 增加 SandboxFusion 服务资源

---

## 关键文件索引

| 功能 | 文件路径 |
|------|----------|
| 启动脚本 | `verl/examples/ppo_trainer/run_deepseek7b_llm_sandbox_fusion.sh` |
| 配置定义 | `verl/workers/config/reward_model.py` |
| 奖励加载 | `verl/trainer/ppo/reward.py` |
| Prime 管理器 | `verl/workers/reward_manager/prime.py` |
| 注册系统 | `verl/workers/reward_manager/registry.py` |
| 评分路由 | `verl/utils/reward_score/__init__.py` |
| Sandbox API | `verl/utils/reward_score/sandbox_fusion/__init__.py` |
| API 实现 | `verl/utils/reward_score/sandbox_fusion/utils.py` |

---

## 相关链接

- **veRL 项目**: https://github.com/volcengine/verl
- **SandboxFusion**: https://github.com/bytedance/SandboxFusion
- **SandboxFusion 文档**: https://bytedance.github.io/SandboxFusion/
