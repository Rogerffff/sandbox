# SandboxFusion 使用指南

## 项目简介

**SandboxFusion** 是由字节跳动开发的安全代码沙盒系统，专为运行和评测 LLM（大语言模型）生成的代码而设计。它是 FullStack Bench 项目的核心组件，支持多种编程语言和代码执行模式。

### 核心特性

- **多语言支持**：支持 30+ 种编程语言，包括 Python、C++、Go、Java、Node.js、TypeScript 等
- **安全隔离**：提供轻量级隔离模式（OverlayFS + cgroups）和资源限制
- **Jupyter 支持**：支持 Jupyter Notebook 多 cell 执行
- **数据集评估**：集成 13+ 个标准代码评估数据集（HumanEval、MBPP、PAL-Math 等）
- **文件操作**：支持代码执行时的文件上传和下载
- **异步执行**：基于 FastAPI 和 asyncio 的高性能异步执行

### 支持的编程语言

| 类型 | 语言 |
|------|------|
| 编译型语言 | C++, Go, Java, C#, Rust, Kotlin, Swift, Scala |
| 脚本语言 | Python, Node.js, TypeScript, PHP, Ruby, Lua, R, Perl, Julia |
| 测试框架 | pytest, Jest, JUnit, go test |
| 特殊类型 | Jupyter, CUDA, Verilog, Lean, SQL, Bash |

---

## 快速开始

### 1. 安装 Python SDK

```bash
pip install sandbox-fusion
```

### 2. 配置 API 端点

```python
from sandbox_fusion import set_endpoint

# 设置你的 SandboxFusion 服务端点
set_endpoint("http://localhost:8080")
```

### 3. 运行代码

```python
from sandbox_fusion import run_code, RunCodeRequest

# 执行 Python 代码
response = run_code(RunCodeRequest(
    code='print("Hello, SandboxFusion!")',
    language='python'
))

print(response.run_result.stdout)  # 输出: Hello, SandboxFusion!
```

### 4. 使用数据集评估

```python
from sandbox_fusion import submit, SubmitRequest, TestConfig

# 提交 HumanEval 评估
result = submit(SubmitRequest(
    dataset='humaneval',
    id='0',
    completion='def solution(x):\n    return x + 1',
    config=TestConfig(language='python')
))

print(f"是否通过: {result.accepted}")
```

---

## 文档目录

| 文档 | 说明 |
|------|------|
| [01-安装部署.md](01-安装部署.md) | 本地开发和 Docker 部署指南 |
| [02-Python-SDK使用指南.md](02-Python-SDK使用指南.md) | Python SDK 完整使用教程 |
| [03-运行代码.md](03-运行代码.md) | 代码执行详解，包括 Jupyter 模式和文件操作 |
| [04-使用数据集.md](04-使用数据集.md) | 评估数据集使用指南 |
| [05-API参考.md](05-API参考.md) | REST API 和数据模型参考 |
| [06-veRL框架集成指南.md](06-veRL框架集成指南.md) | **veRL 强化学习框架集成** - RL 训练中使用 SandboxFusion |

---

## 项目结构

```
SandboxFusion/
├── sandbox/                    # 核心代码库
│   ├── runners/               # 代码执行引擎（支持多语言）
│   ├── datasets/              # 评估数据集实现
│   ├── server/                # FastAPI Web 服务
│   ├── configs/               # 配置管理
│   └── utils/                 # 工具函数库
├── runtime/                    # 语言运行时环境
│   ├── python/                # Python 运行时
│   ├── go/                    # Go 运行时
│   ├── java/                  # Java 依赖
│   ├── node/                  # Node.js 运行时
│   ├── jupyter/               # Jupyter 内核
│   └── lean/                  # Lean 证明助手
├── scripts/                    # 构建和部署脚本
│   ├── client/                # Python 客户端 SDK
│   └── Dockerfile.*           # Docker 构建文件
└── docs/                       # 文档网站
```

---

## 相关链接

- **项目地址**: https://github.com/bytedance/SandboxFusion
- **官方文档**: https://bytedance.github.io/SandboxFusion/
- **PyPI**: https://pypi.org/project/sandbox-fusion/
