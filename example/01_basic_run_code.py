"""
SandboxFusion 示例 1: 基础代码执行
=====================================
本示例展示如何使用 SandboxFusion 执行基本的 Python、C++ 和 Go 代码。

使用前请确保:
1. SandboxFusion 服务正在运行 (make run)
2. 已安装 sandbox-fusion SDK (pip install sandbox-fusion)
"""

from sandbox_fusion import run_code, RunCodeRequest, set_endpoint

# 设置 API 端点（默认为 localhost:8080）
set_endpoint("http://localhost:8080")


def run_python_example():
    """执行 Python 代码示例"""
    print("=" * 50)
    print("示例 1.1: 执行 Python 代码")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
def fibonacci(n):
    """计算斐波那契数列"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 计算前 10 个斐波那契数
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
''',
        language='python',
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    if response.run_result.stderr:
        print(f"错误: {response.run_result.stderr}")
    print()


def run_python_with_input():
    """执行带标准输入的 Python 代码"""
    print("=" * 50)
    print("示例 1.2: Python 代码（带标准输入）")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
name = input("请输入你的名字: ")
age = input("请输入你的年龄: ")
print(f"你好 {name}，你今年 {age} 岁了！")
''',
        language='python',
        stdin='张三\n25\n',  # 模拟用户输入
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    print()


def run_cpp_example():
    """执行 C++ 代码示例"""
    print("=" * 50)
    print("示例 1.3: 执行 C++ 代码")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> numbers = {5, 2, 8, 1, 9, 3};

    cout << "排序前: ";
    for (int n : numbers) cout << n << " ";
    cout << endl;

    sort(numbers.begin(), numbers.end());

    cout << "排序后: ";
    for (int n : numbers) cout << n << " ";
    cout << endl;

    return 0;
}
''',
        language='cpp',
        compile_timeout=30,
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    if response.compile_result:
        print(f"编译状态: {response.compile_result.status}")
    print(f"输出:\n{response.run_result.stdout}")
    if response.run_result.stderr:
        print(f"错误: {response.run_result.stderr}")
    print()


def run_go_example():
    """执行 Go 代码示例"""
    print("=" * 50)
    print("示例 1.4: 执行 Go 代码")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
package main

import (
    "fmt"
    "strings"
)

func main() {
    words := []string{"Hello", "SandboxFusion", "World"}

    // 将单词连接成句子
    sentence := strings.Join(words, " ")
    fmt.Println(sentence)

    // 字符串反转
    runes := []rune(sentence)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    fmt.Println("反转后:", string(runes))
}
''',
        language='go',
        compile_timeout=30,
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    if response.run_result.stderr:
        print(f"错误: {response.run_result.stderr}")
    print()


def run_with_timeout_example():
    """演示超时处理"""
    print("=" * 50)
    print("示例 1.5: 超时处理示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
import time
print("开始执行...")
time.sleep(5)  # 等待 5 秒
print("执行完成")
''',
        language='python',
        run_timeout=2  # 设置 2 秒超时
    ))

    print(f"状态: {response.status}")
    print(f"运行状态: {response.run_result.status}")
    if response.run_result.status == 'TimeLimitExceeded':
        print("代码执行超时！")
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("   SandboxFusion 基础代码执行示例")
    print("=" * 60 + "\n")

    try:
        run_python_example()
        run_python_with_input()
        run_cpp_example()
        run_go_example()
        run_with_timeout_example()

        print("=" * 60)
        print("   所有示例执行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保 SandboxFusion 服务正在运行 (make run)")


if __name__ == "__main__":
    main()
