# Bug 修复：StreamWriter stdin 写入问题

## 问题概述

**问题编号**: BUG-2026-0204  
**影响版本**: SandboxFusion 原始版本  
**修复日期**: 2026-02-04  
**严重程度**: 高（导致 CodeContests 等需要 stdin 输入的评测完全失败）

## 问题现象

在评测 CodeContests 数据集时，sandbox 日志反复出现以下错误：

```
AttributeError: 'StreamWriter' object has no attribute 'flush'
```

**具体表现**：
- HumanEval、MBPP 数据集评测正常
- CodeContests 数据集评测极慢，几乎不动
- 每个测试用例都因 stdin 写入失败而等待超时（30秒）
- 所有 CodeContests 测试结果都是失败

## 根本原因

### 问题代码位置

`sandbox/runners/base.py` 第 69-82 行：

```python
# 原始代码（错误）
p = await asyncio.create_subprocess_shell(...)  # 创建异步子进程
if stdin is not None:
    if p.stdin:
        p.stdin.write(stdin.encode())
        p.stdin.flush()   # ❌ 错误！StreamWriter 没有 flush() 方法
```

### 技术原因：同步 vs 异步 API 混淆

Python 中创建子进程有两种方式，它们的 `stdin` 对象类型不同：

| 创建方式 | stdin 类型 | 刷新缓冲区方法 |
|---------|-----------|---------------|
| `subprocess.Popen()` (同步) | `io.BufferedWriter` | `stdin.flush()` (同步方法) |
| `asyncio.create_subprocess_*()` (异步) | `asyncio.StreamWriter` | `await stdin.drain()` (协程) |

sandbox 使用的是 **异步** 方式创建子进程，但错误地使用了同步 API 的 `flush()` 方法。

### `asyncio.StreamWriter` 的正确用法

```python
# StreamWriter 的方法
writer.write(data)      # 同步，写入缓冲区（不等待发送）
await writer.drain()    # 协程，等待缓冲区数据发送到管道
writer.close()          # 同步，关闭流
await writer.wait_closed()  # 协程，等待流完全关闭
```

**注意**：`StreamWriter` 没有 `flush()` 方法，必须使用 `await drain()` 来确保数据发送。

## 为什么只影响 CodeContests？

不同数据集的测试方式不同：

| 数据集 | 测试方式 | 是否需要 stdin | 是否触发 bug |
|--------|---------|---------------|-------------|
| HumanEval | Python assert 语句 | ❌ 不需要 | ✅ 正常 |
| MBPP | Python assert 语句 | ❌ 不需要 | ✅ 正常 |
| CodeContests | stdin/stdout 比对 | ✅ 需要 | ❌ 触发 bug |

CodeContests 是编程竞赛题，程序需要：
1. 从 `stdin` 读取输入数据
2. 计算结果
3. 输出到 `stdout`

由于 `flush()` 调用失败抛出异常，输入数据无法传递给子进程，程序一直等待输入直到超时。

## 修复方案

### 修改文件

`sandbox/runners/base.py`

### 修改内容

```python
# 修复前（错误）
if stdin is not None:
    try:
        if p.stdin:
            p.stdin.write(stdin.encode())
            p.stdin.flush()  # ❌ StreamWriter 没有此方法
        else:
            logger.warning("Attempted to write to stdin, but stdin is closed.")
    except Exception as e:
        logger.exception(f"Failed to write to stdin: {e}")
if p.stdin:
    try:
        p.stdin.close()
    except Exception as e:
        logger.warning(f"Failed to close stdin: {e}")

# 修复后（正确）
if stdin is not None:
    try:
        if p.stdin:
            p.stdin.write(stdin.encode())
            await p.stdin.drain()  # ✅ 使用 await drain() 替代 flush()
        else:
            logger.warning("Attempted to write to stdin, but stdin is closed.")
    except Exception as e:
        logger.exception(f"Failed to write to stdin: {e}")
if p.stdin:
    try:
        p.stdin.close()
        await p.stdin.wait_closed()  # ✅ 等待 stdin 完全关闭
    except Exception as e:
        logger.warning(f"Failed to close stdin: {e}")
```

### 关键改动

1. **`p.stdin.flush()` → `await p.stdin.drain()`**
   - `drain()` 是协程，必须使用 `await`
   - 它会暂停执行，直到写入缓冲区的数据被发送到管道

2. **添加 `await p.stdin.wait_closed()`**（推荐）
   - 确保 stdin 流被完全关闭后再继续
   - 防止潜在的竞态条件

## 验证修复

### 测试命令

```bash
# 测试 stdin 输入功能
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import sys\ndata = sys.stdin.read().split()\nprint(int(data[0]) + int(data[1]))",
    "language": "python",
    "run_timeout": 10,
    "stdin": "3 5"
  }'
```

### 预期结果

```json
{
    "status": "Success",
    "run_result": {
        "status": "Finished",
        "stdout": "8\n",
        "stderr": ""
    }
}
```

### 修复前的错误结果

```json
{
    "status": "SandboxError",
    "message": "Failed to write to stdin: 'StreamWriter' object has no attribute 'flush'"
}
```

## 影响范围

### 受影响的功能

- `run_code` API 的 `stdin` 参数
- 所有需要 stdin 输入的代码执行
- CodeContests、ACM 等竞赛题评测

### 不受影响的功能

- 不使用 stdin 的代码执行
- HumanEval、MBPP 等使用 assert 测试的数据集
- `submit` API（使用内置测试用例）

## 经验总结

### 避免类似问题的建议

1. **区分同步和异步 API**
   - `subprocess` 模块是同步的
   - `asyncio.subprocess` 模块是异步的
   - 两者的对象接口相似但不完全相同

2. **查阅官方文档**
   - [asyncio.StreamWriter 文档](https://docs.python.org/3/library/asyncio-stream.html#asyncio.StreamWriter)
   - 注意方法是同步还是协程

3. **编写单元测试**
   - 为 stdin 输入场景编写测试用例
   - 确保竞赛题评测流程被测试覆盖

## 参考资料

- [Python asyncio.subprocess 官方文档](https://docs.python.org/3/library/asyncio-subprocess.html)
- [asyncio.StreamWriter API 参考](https://docs.python.org/3/library/asyncio-stream.html#asyncio.StreamWriter)
- [subprocess vs asyncio.subprocess 对比](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio-subprocess)
