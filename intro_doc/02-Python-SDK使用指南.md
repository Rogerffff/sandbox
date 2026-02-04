# Python SDK 使用指南

本文档详细介绍 SandboxFusion Python SDK 的使用方法。

---

## 安装

```bash
pip install sandbox-fusion
```

---

## 配置 API 端点

在使用 SDK 之前，需要配置 SandboxFusion 服务的端点地址。有三种方式：

### 方式一：环境变量（推荐）

```bash
export SANDBOX_FUSION_ENDPOINT="http://your-api-endpoint.com"
```

### 方式二：函数设置

```python
from sandbox_fusion import set_endpoint

set_endpoint("http://localhost:8080")
```

### 方式三：函数参数

在单个函数调用时直接指定：

```python
response = run_code(request, endpoint="http://localhost:8080")
```

---

## 核心 API 函数

### 函数概览

| 函数 | 说明 | 自动重试 |
|------|------|---------|
| `run_code()` | 运行代码脚本 | 5 次 |
| `run_jupyter()` | 运行 Jupyter 笔记本 | 3 次 |
| `get_prompts()` | 获取数据集提示列表 | 无 |
| `get_prompt_by_id()` | 按 ID 获取提示 | 无 |
| `submit()` | 提交评估请求 | 5 次 |
| `submit_safe()` | 安全提交（失败不抛异常） | 5 次 |

---

### run_code() - 运行代码

执行单个代码文件并返回结果。

```python
from sandbox_fusion import run_code, RunCodeRequest

# 基础用法
response = run_code(RunCodeRequest(
    code='print("Hello, World!")',
    language='python'
))

# 检查结果
if response.status == 'Success':
    print(f"输出: {response.run_result.stdout}")
else:
    print(f"错误: {response.run_result.stderr}")
```

#### 完整参数

```python
request = RunCodeRequest(
    code='...',                    # 代码字符串（必需）
    language='python',             # 编程语言（必需）
    compile_timeout=10,            # 编译超时（秒）
    run_timeout=10,                # 运行超时（秒）
    memory_limit_MB=-1,            # 内存限制（MB，-1 表示无限制）
    stdin='input data',            # 标准输入
    files={'data.txt': 'base64...'}, # 上传文件（base64 编码）
    fetch_files=['output.txt']     # 执行后获取的文件
)
```

---

### run_jupyter() - 运行 Jupyter

执行多个 Jupyter cell 并返回每个 cell 的结果。

```python
from sandbox_fusion import run_jupyter, RunJupyterRequest

response = run_jupyter(RunJupyterRequest(
    cells=[
        'import numpy as np',
        'a = np.array([1, 2, 3])',
        'print(a.sum())'
    ],
    cell_timeout=10,       # 单个 cell 超时
    total_timeout=45,      # 总超时
    kernel='python3'       # 内核类型
))

# 遍历每个 cell 的结果
for i, cell_result in enumerate(response.cells):
    print(f"Cell {i}:")
    print(f"  stdout: {cell_result.stdout}")
    print(f"  stderr: {cell_result.stderr}")
    print(f"  display: {cell_result.display}")
    print(f"  error: {cell_result.error}")
```

---

### submit() - 提交评估

向数据集提交代码进行评估。

```python
from sandbox_fusion import submit, SubmitRequest, TestConfig

result = submit(SubmitRequest(
    dataset='humaneval',           # 数据集名称
    id='0',                        # 问题 ID
    completion='def solution(x):\n    return x + 1',  # 代码完成
    config=TestConfig(
        language='python',
        compile_timeout=10,
        run_timeout=20
    )
))

# 检查评估结果
print(f"是否通过: {result.accepted}")
print(f"提取的代码: {result.extracted_code}")
print(f"测试结果: {result.tests}")
```

---

### submit_safe() - 安全提交

与 `submit()` 类似，但失败时返回拒绝结果而不是抛出异常。

```python
from sandbox_fusion import submit_safe

result = submit_safe(request)

if result.accepted:
    print("评估通过")
else:
    print(f"评估失败或出错")
```

---

### get_prompts() - 获取提示列表

获取数据集中的提示列表。

```python
from sandbox_fusion import get_prompts, GetPromptsRequest

prompts = get_prompts(GetPromptsRequest(
    dataset='humaneval',
    config={'language': 'python'}
))

for prompt in prompts:
    print(f"ID: {prompt.id}")
    print(f"提示: {prompt.prompt[:100]}...")
```

---

### get_prompt_by_id() - 按 ID 获取提示

获取特定 ID 的提示。

```python
from sandbox_fusion import get_prompt_by_id, GetPromptByIdRequest

prompt = get_prompt_by_id(GetPromptByIdRequest(
    dataset='humaneval',
    id='0',
    config={'language': 'python'}
))

print(f"提示内容: {prompt.prompt}")
```

---

## 异步接口

所有 API 函数都有对应的异步版本，只需添加 `_async` 后缀：

```python
import asyncio
from sandbox_fusion import run_code_async, RunCodeRequest

async def main():
    response = await run_code_async(RunCodeRequest(
        code='print("async hello")',
        language='python'
    ))
    print(response.run_result.stdout)

asyncio.run(main())
```

可用的异步函数：
- `run_code_async()`
- `run_jupyter_async()`
- `submit_async()`
- `submit_safe_async()`
- `get_prompts_async()`
- `get_prompt_by_id_async()`

---

## 并发执行

使用 `run_concurrent()` 批量执行多个请求：

```python
from sandbox_fusion import run_concurrent, run_code, RunCodeRequest

# 准备多个请求
requests = [
    RunCodeRequest(code=f'print({i})', language='python')
    for i in range(10)
]

# 并发执行
results = run_concurrent(
    func=run_code,
    args_list=requests,
    max_workers=5  # 最大并发数
)

for i, result in enumerate(results):
    print(f"请求 {i}: {result.run_result.stdout}")
```

---

## 超时设置

可以为请求设置客户端超时：

```python
from sandbox_fusion import run_code, RunCodeRequest

response = run_code(
    RunCodeRequest(
        code='import time; time.sleep(30)',
        language='python',
        run_timeout=60  # 服务端超时
    ),
    client_timeout=120  # 客户端超时（秒）
)
```

---

## 完整示例

### 示例 1：执行 Python 代码并获取文件

```python
from sandbox_fusion import run_code, RunCodeRequest
import base64

# 执行代码并获取生成的文件
response = run_code(RunCodeRequest(
    code='''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sin Wave')
plt.savefig('plot.png')
print('Plot saved!')
''',
    language='python',
    fetch_files=['plot.png']
))

# 保存获取的文件
if 'plot.png' in response.files:
    with open('downloaded_plot.png', 'wb') as f:
        f.write(base64.b64decode(response.files['plot.png']))
    print('文件已保存')
```

### 示例 2：批量评估 HumanEval

```python
from sandbox_fusion import (
    get_prompts, submit,
    GetPromptsRequest, SubmitRequest, TestConfig
)

# 获取所有提示
prompts = get_prompts(GetPromptsRequest(
    dataset='humaneval',
    config={'language': 'python'}
))

# 模拟代码生成（实际场景中由 LLM 生成）
def generate_code(prompt):
    # 这里应该调用你的 LLM
    return "def solution(x):\n    pass"

# 评估每个问题
results = []
for prompt in prompts[:5]:  # 只测试前 5 个
    completion = generate_code(prompt.prompt)

    result = submit(SubmitRequest(
        dataset='humaneval',
        id=prompt.id,
        completion=completion,
        config=TestConfig(language='python')
    ))

    results.append({
        'id': prompt.id,
        'accepted': result.accepted
    })
    print(f"问题 {prompt.id}: {'通过' if result.accepted else '失败'}")

# 计算通过率
pass_rate = sum(r['accepted'] for r in results) / len(results)
print(f"\n通过率: {pass_rate:.2%}")
```

### 示例 3：运行 Jupyter 并捕获图表

```python
from sandbox_fusion import run_jupyter, RunJupyterRequest
import base64

response = run_jupyter(RunJupyterRequest(
    cells=[
        'import matplotlib.pyplot as plt',
        'import numpy as np',
        'x = np.random.randn(1000)',
        'plt.hist(x, bins=30)',
        'plt.title("Random Distribution")',
        'plt.show()'
    ],
    cell_timeout=30,
    total_timeout=60
))

# 检查是否有图表输出
for i, cell in enumerate(response.cells):
    if cell.display:
        print(f"Cell {i} 有图表输出")
        # display 中包含 base64 编码的图片
```

---

## 下一步

- [03-运行代码.md](03-运行代码.md) - 了解更多代码执行细节
- [04-使用数据集.md](04-使用数据集.md) - 学习各种数据集的使用方法
- [05-API参考.md](05-API参考.md) - 查看完整的 API 参考
