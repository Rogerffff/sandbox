# SandboxFusion 代码库导航

SandboxFusion 是字节跳动开发的安全代码沙盒系统，用于执行和评估 LLM 生成的代码，支持 30+ 编程语言，集成 15+ 编码基准数据集，主要服务于 RL 训练中的代码评估和奖励计算。

## 目录结构

```
SandboxFusion/
├── sandbox/           # 核心应用代码
│   ├── server/        # FastAPI REST API 服务
│   ├── runners/       # 各语言代码执行引擎
│   ├── datasets/      # 评测数据集实现
│   ├── configs/       # 配置管理
│   ├── utils/         # 工具函数
│   ├── pages/         # Web UI
│   ├── tests/         # 单元测试
│   ├── database.py    # 数据库操作 (MySQL/SQLite)
│   └── registry.py    # 数据集注册表
├── runtime/           # 各语言运行时安装脚本和依赖
├── scripts/           # Docker 构建和部署脚本
├── example/           # 使用示例 (4个)
├── docs/              # Docusaurus 文档站
├── intro_doc/         # 中文文档
├── Makefile           # 构建/测试/格式化命令
└── pyproject.toml     # Poetry 依赖配置
```

## 核心模块详解

### sandbox/runners/ — 代码执行引擎

| 文件 | 内容 |
|------|------|
| `types.py` | 核心数据模型: `CodeRunArgs`, `CodeRunResult`, `Language` 类型定义 |
| `base.py` | 底层命令执行: `run_command_bare()`, `run_commands()`, 隔离模式(none/lite) |
| `major.py` | 主要语言: Python, C++, Java, Go, C#, Node.js, TypeScript 等 |
| `minor.py` | 小众语言: Lua, R, Perl, D, Ruby, Scala, Julia, Kotlin, Rust, PHP, Verilog, Lean, Swift, Racket |
| `cuda.py` | GPU 执行: CUDA, Python GPU |
| `jupyter.py` | Jupyter notebook 多 cell 执行 |
| `isolation.py` | 进程隔离: overlayfs, cgroups, 网络命名空间 |

### sandbox/datasets/ — 评测数据集

| 文件 | 数据集 | 说明 |
|------|--------|------|
| `types.py` | — | 基类 `CodingDataset`, `Prompt`, `EvalResult` 等 |
| `humaneval.py` | HumanEval | 164 题，支持 22+ 语言 |
| `mbpp.py` | MBPP | 500 道 Python 基础题 |
| `live_code_bench.py` | LiveCodeBench | 402 题，多代码风格 (最大实现，1052行) |
| `codecontests.py` | CodeContests | 竞赛编程 |
| `cruxeval.py` | CRUXEval | 代码理解评测 |
| `natural_code_bench.py` | NaturalCodeBench | 真实编程任务 |
| `palmath.py` | PAL-Math | 数学推理 |
| `common_oj.py` | — | 在线判题系统公共基类 |
| `mbxp.py` | MBXP | 多语言 MBPP |
| `verilog.py` | Verilog | 硬件描述 |
| `mhpp.py` | MHPP | 混合问题 |
| `repobench_c.py` / `repobench_p.py` | RepoBench | 仓库级代码检索 |

### sandbox/server/ — API 服务

| 文件 | 端点 | 说明 |
|------|------|------|
| `server.py` | — | FastAPI 应用初始化、路由注册 |
| `sandbox_api.py` | `POST /run_code`, `POST /run_jupyter` | 代码执行 API |
| `online_judge_api.py` | `GET /list_datasets`, `POST /get_prompts`, `POST /submit`, `POST /get_metrics` | 数据集评测 API |

### sandbox/utils/ — 工具函数

| 文件 | 用途 |
|------|------|
| `extraction.py` | 从 LLM 输出中提取代码 (围栏代码块、启发式提取) |
| `sandbox_client.py` | Python SDK 客户端: `run_code()`, `submit()`, `run_concurrent()` |
| `testing.py` | 测试工具和断言检查 |
| `execution.py` | 进程管理 |
| `common.py` | 通用工具 (权限、conda、路径) |
| `antihack.py` | 安全防护措施 |
| `prompting.py` | Prompt 格式化 |

### sandbox/configs/ — 配置

- `run_config.py`: `RunConfig`, `SandboxConfig`, `DatasetConfig` 配置类
- `local.yaml`: 开发环境 (`isolation: none`)
- `ci.yaml`: CI/生产环境 (`isolation: lite`)

## runtime/ — 运行时环境

| 目录 | 内容 |
|------|------|
| `python/` | conda 环境安装脚本、requirements.txt、PyTorch/GPU 安装 |
| `java/` | JUnit JAR 依赖 |
| `go/` | Go 模块模板 |
| `node/` | Node.js 依赖 (babel, jest) |
| `jupyter/` | Jupyter kernel 实现 |
| `lean/` | Lean 定理证明器 |

## example/ — 使用示例

1. `01_basic_run_code.py` — 基础代码执行
2. `02_multi_language.py` — 多语言支持
3. `03_dataset_for_rl_training.py` — 数据集 API 用于 RL 训练
4. `04_rl_reward_integration.py` — 完整 RL 奖励集成 (`RewardCalculator`, 批量评估)

## 常用命令 (Makefile)

- `make run` — 启动开发服务器 (端口 8080，自动重载)
- `make test` — 运行测试 (4 worker 并行)
- `make format` — 代码格式化 (yapf, isort, pycln)

## RL 训练工作流

1. `get_prompts()` → 获取题目作为模型输入
2. 模型生成代码
3. `submit()` → 执行测试用例，返回 pass/fail
4. `EvalResult.accepted` → 作为奖励信号
5. `run_concurrent()` → 批量并行评估
