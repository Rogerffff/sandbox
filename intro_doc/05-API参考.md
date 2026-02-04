# API 参考

本文档提供 SandboxFusion 的完整 API 参考，包括 REST API 端点和数据模型定义。

---

## REST API 端点

### 代码执行 API

#### POST /run_code

执行单个代码文件。

**请求体**：

```json
{
    "code": "print('hello')",
    "language": "python",
    "compile_timeout": 10,
    "run_timeout": 10,
    "memory_limit_MB": -1,
    "stdin": "",
    "files": {},
    "fetch_files": []
}
```

**响应体**：

```json
{
    "status": "Success",
    "compile_result": null,
    "run_result": {
        "status": "Finished",
        "execution_time": 0.05,
        "return_code": 0,
        "stdout": "hello\n",
        "stderr": ""
    },
    "files": {}
}
```

---

#### POST /run_jupyter

执行 Jupyter Notebook 多个单元格。

**请求体**：

```json
{
    "cells": ["import numpy", "np.array([1,2,3])"],
    "cell_timeout": 10,
    "total_timeout": 45,
    "memory_limit_MB": -1,
    "kernel": "python3"
}
```

**响应体**：

```json
{
    "cells": [
        {
            "stdout": "",
            "stderr": "",
            "display": null,
            "error": null
        },
        {
            "stdout": "",
            "stderr": "",
            "display": "array([1, 2, 3])",
            "error": null
        }
    ]
}
```

---

### 数据集评估 API

#### POST /get_prompts

获取数据集的提示列表。

**请求体**：

```json
{
    "dataset": "humaneval",
    "config": {
        "language": "python"
    }
}
```

**响应体**：

```json
{
    "prompts": [
        {
            "id": "0",
            "prompt": "from typing import List\n\ndef has_close_elements..."
        }
    ]
}
```

---

#### POST /get_prompt_by_id

按 ID 获取特定提示。

**请求体**：

```json
{
    "dataset": "humaneval",
    "id": "0",
    "config": {
        "language": "python"
    }
}
```

**响应体**：

```json
{
    "id": "0",
    "prompt": "from typing import List\n\ndef has_close_elements..."
}
```

---

#### POST /submit

提交代码进行评估。

**请求体**：

```json
{
    "dataset": "humaneval",
    "id": "0",
    "completion": "def solution(x):\n    return x + 1",
    "config": {
        "language": "python",
        "compile_timeout": 10,
        "run_timeout": 20
    }
}
```

**响应体**：

```json
{
    "accepted": true,
    "extracted_code": "def solution(x):\n    return x + 1",
    "tests": [
        {
            "status": "Passed",
            "stdout": "",
            "stderr": ""
        }
    ],
    "extra": {}
}
```

---

#### POST /get_metrics

获取数据集的评估指标。

**请求体**：

```json
{
    "dataset": "humaneval",
    "results": [...]
}
```

---

## 数据模型定义

### 代码执行相关

#### RunCodeRequest

```python
class RunCodeRequest(BaseModel):
    code: str                          # 代码字符串（必需）
    language: Language                 # 编程语言（必需）
    compile_timeout: float = 10        # 编译超时（秒）
    run_timeout: float = 10            # 运行超时（秒）
    memory_limit_MB: int = -1          # 内存限制（MB，-1 无限制）
    stdin: Optional[str] = None        # 标准输入
    files: Dict[str, str] = {}         # 上传文件（路径 -> base64 内容）
    fetch_files: List[str] = []        # 执行后获取的文件路径
```

#### RunCodeResponse

```python
class RunCodeResponse(BaseModel):
    status: RunStatus                  # 执行状态
    compile_result: Optional[CommandRunResult]  # 编译结果
    run_result: Optional[CommandRunResult]      # 运行结果
    files: Dict[str, str] = {}         # 获取的文件（路径 -> base64 内容）
```

#### CommandRunResult

```python
class CommandRunResult(BaseModel):
    status: CommandRunStatus           # 命令状态
    execution_time: Optional[float]    # 执行时间（秒）
    return_code: Optional[int]         # 返回码
    stdout: Optional[str]              # 标准输出
    stderr: Optional[str]              # 标准错误
```

#### RunStatus

```python
RunStatus = Literal['Success', 'Failed', 'SandboxError']
```

| 值 | 说明 |
|----|------|
| `Success` | 执行成功（返回码为 0） |
| `Failed` | 执行失败（返回码非 0 或超时） |
| `SandboxError` | 沙盒内部错误 |

#### CommandRunStatus

```python
CommandRunStatus = Literal['Finished', 'Error', 'TimeLimitExceeded']
```

| 值 | 说明 |
|----|------|
| `Finished` | 命令正常完成 |
| `Error` | 命令执行出错 |
| `TimeLimitExceeded` | 超时 |

---

### Jupyter 相关

#### RunJupyterRequest

```python
class RunJupyterRequest(BaseModel):
    cells: List[str]                   # 代码单元格列表
    cell_timeout: float = 10           # 单个 cell 超时（秒）
    total_timeout: float = 45          # 总超时（秒）
    memory_limit_MB: int = -1          # 内存限制（MB）
    kernel: str = 'python3'            # 内核类型
```

#### RunJupyterResponse

```python
class RunJupyterResponse(BaseModel):
    cells: List[CellRunResult]         # 每个 cell 的执行结果
```

#### CellRunResult

```python
class CellRunResult(BaseModel):
    stdout: Optional[str]              # 标准输出
    stderr: Optional[str]              # 标准错误
    display: Optional[Any]             # 富文本输出（图表、DataFrame 等）
    error: Optional[str]               # 异常信息（traceback）
```

---

### 数据集评估相关

#### GetPromptsRequest

```python
class GetPromptsRequest(BaseModel):
    dataset: str                       # 数据集名称
    config: Dict[str, Any] = {}        # 数据集配置
```

#### GetPromptByIdRequest

```python
class GetPromptByIdRequest(BaseModel):
    dataset: str                       # 数据集名称
    id: str                            # 问题 ID
    config: Dict[str, Any] = {}        # 数据集配置
```

#### Prompt

```python
class Prompt(BaseModel):
    id: str                            # 问题 ID
    prompt: str                        # 提示内容
```

#### SubmitRequest

```python
class SubmitRequest(BaseModel):
    dataset: str                       # 数据集名称
    id: str                            # 问题 ID
    completion: str                    # 代码完成/答案
    config: TestConfig                 # 测试配置
```

#### TestConfig

```python
class TestConfig(BaseModel):
    language: Optional[Language] = None      # 编程语言
    compile_timeout: float = 10              # 编译超时
    run_timeout: float = 20                  # 运行超时
    dataset_type: Optional[str] = None       # 数据集类型
    provided_data: Optional[Dict] = None     # 自定义数据
```

#### EvalResult

```python
class EvalResult(BaseModel):
    accepted: bool                     # 是否通过
    extracted_code: Optional[str]      # 提取的代码
    tests: Optional[List[EvalTestCase]]  # 测试用例结果
    extra: Dict[str, Any] = {}         # 额外信息
```

#### EvalTestCase

```python
class EvalTestCase(BaseModel):
    status: str                        # 测试状态
    stdout: Optional[str]              # 标准输出
    stderr: Optional[str]              # 标准错误
```

---

## 语言枚举值

```python
Language = Literal[
    # 编译型语言
    'cpp',              # C++
    'go',               # Go
    'go_test',          # Go 测试模式
    'java',             # Java
    'junit',            # Java JUnit 测试
    'csharp',           # C#
    'rust',             # Rust
    'kotlin_script',    # Kotlin 脚本
    'swift',            # Swift
    'scala',            # Scala
    'D_ut',             # D 语言

    # 脚本语言
    'python',           # Python
    'pytest',           # Python 测试模式
    'nodejs',           # Node.js
    'typescript',       # TypeScript
    'jest',             # JavaScript 测试模式
    'php',              # PHP
    'ruby',             # Ruby
    'lua',              # Lua
    'R',                # R 语言
    'perl',             # Perl
    'julia',            # Julia
    'racket',           # Racket

    # 特殊类型
    'cuda',             # CUDA
    'python_gpu',       # Python GPU 模式
    'verilog',          # Verilog
    'lean',             # Lean 4
    'sql',              # SQL
    'bash'              # Bash
]
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
    "detail": "错误描述信息"
}
```

### SDK 异常

```python
from sandbox_fusion import SandboxError

try:
    result = run_code(request)
except SandboxError as e:
    print(f"沙盒错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

---

## cURL 示例

### 执行 Python 代码

```bash
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"Hello, World!\")",
    "language": "python",
    "run_timeout": 10
  }'
```

### 执行 Jupyter

```bash
curl -X POST http://localhost:8080/run_jupyter \
  -H "Content-Type: application/json" \
  -d '{
    "cells": ["a = 1", "print(a + 1)"],
    "cell_timeout": 10,
    "total_timeout": 30
  }'
```

### 提交 HumanEval 评估

```bash
curl -X POST http://localhost:8080/submit \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "humaneval",
    "id": "0",
    "completion": "def solution(x):\n    return x + 1",
    "config": {
      "language": "python",
      "run_timeout": 20
    }
  }'
```

### 获取数据集提示

```bash
curl -X POST http://localhost:8080/get_prompts \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "humaneval",
    "config": {"language": "python"}
  }'
```

---

## 配置参数汇总

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compile_timeout` | float | 10 | 编译超时（秒） |
| `run_timeout` | float | 10/20 | 运行超时（秒） |
| `memory_limit_MB` | int | -1 | 内存限制（-1 无限制） |

### 数据集特定参数

| 数据集 | 参数 | 说明 |
|--------|------|------|
| HumanEval | `is_freeform` | 指令微调模式 |
| HumanEval | `locale` | 提示语言（en/zh） |
| MBPP | `is_fewshot` | 是否添加示例 |
| CRUXEval | `mode` | input 或 output |
| CRUXEval | `coding_wrap_prompt` | 聊天格式包装 |
| CRUXEval | `use_cot` | 思维链指导 |
| PAL-Math | `subset` | 子数据集名称 |
| Code Contests | `locale` | 题目语言 |

---

## 相关链接

- **项目 GitHub**: https://github.com/bytedance/SandboxFusion
- **官方文档**: https://bytedance.github.io/SandboxFusion/
- **PyPI 包**: https://pypi.org/project/sandbox-fusion/
