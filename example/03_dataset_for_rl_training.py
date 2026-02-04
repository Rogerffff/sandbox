"""
SandboxFusion 示例 3: RL 训练 Coding Model 的数据集使用
=========================================================
本示例详细展示在强化学习训练代码生成模型时，如何使用 SandboxFusion：

1. 获取数据集的 prompts（作为模型输入）
2. 提交模型生成的代码进行评测
3. 获取评测结果作为 reward 信号
4. 批量评测和并发处理

支持的数据集：
- HumanEval: OpenAI 的代码生成基准 (164 题)
- MBPP: Google 的基础 Python 编程问题
- PAL-Math: 数学推理问题
- Code Contests: 竞赛编程问题
- 等等...
"""

from sandbox_fusion import (
    get_prompts,
    get_prompt_by_id,
    submit,
    submit_safe,
    run_code,
    run_concurrent,
    GetPromptsRequest,
    GetPromptByIdRequest,
    SubmitRequest,
    RunCodeRequest,
    TestConfig,
    set_endpoint
)
from typing import List, Dict, Any
import json

set_endpoint("http://localhost:8080")


# =============================================================================
# 第一部分：获取数据集 Prompts
# =============================================================================

def example_get_humaneval_prompts():
    """
    示例：获取 HumanEval 数据集的所有 prompts
    这些 prompts 可以作为 RL 训练时的模型输入
    """
    print("=" * 60)
    print("示例 1: 获取 HumanEval 数据集 Prompts")
    print("=" * 60)

    # 获取所有 prompts
    prompts = get_prompts(GetPromptsRequest(
        dataset='humaneval',
        config={
            'language': 'python',
            'locale': 'en'  # 可选 'zh' 获取中文提示
        }
    ))

    print(f"总共获取到 {len(prompts)} 个问题\n")

    # 展示前 3 个问题
    for i, prompt in enumerate(prompts[:3]):
        print(f"--- 问题 {prompt.id} ---")
        print(f"Prompt (前 300 字符):\n{prompt.prompt[:300]}...")
        print()

    return prompts


def example_get_single_prompt():
    """
    示例：按 ID 获取单个问题
    在 RL 训练中，可以用于获取特定问题进行采样
    """
    print("=" * 60)
    print("示例 2: 获取单个问题")
    print("=" * 60)

    # 获取 HumanEval 第一个问题
    prompt = get_prompt_by_id(GetPromptByIdRequest(
        dataset='humaneval',
        id='0',
        config={'language': 'python'}
    ))

    print(f"问题 ID: {prompt.id}")
    print(f"完整 Prompt:\n{prompt.prompt}")
    print()

    return prompt


def example_get_mbpp_prompts():
    """
    示例：获取 MBPP 数据集的 prompts
    MBPP 更适合基础编程能力评估
    """
    print("=" * 60)
    print("示例 3: 获取 MBPP 数据集 Prompts")
    print("=" * 60)

    prompts = get_prompts(GetPromptsRequest(
        dataset='mbpp',
        config={
            'is_fewshot': True  # 是否包含 few-shot 示例
        }
    ))

    print(f"MBPP 总共 {len(prompts)} 个问题")
    print(f"\n第一个问题:\n{prompts[0].prompt[:500]}...")
    print()

    return prompts


# =============================================================================
# 第二部分：提交代码评测（获取 Reward）
# =============================================================================

def example_submit_and_get_reward():
    """
    示例：提交模型生成的代码并获取评测结果
    评测结果可以作为 RL 训练的 reward 信号

    返回:
    - accepted: True/False (通过/失败)
    - 可以设计 reward: accepted=1.0, not_accepted=0.0
    """
    print("=" * 60)
    print("示例 4: 提交代码评测（获取 Reward）")
    print("=" * 60)

    # 模拟模型生成的代码（实际场景中由 LLM 生成）
    model_completion = '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """检查列表中是否有两个元素的差值小于给定阈值"""
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
'''

    # 提交评测
    result = submit(SubmitRequest(
        dataset='humaneval',
        id='0',
        completion=model_completion,
        config=TestConfig(
            language='python',
            run_timeout=20
        )
    ))

    print(f"评测结果: {'通过 ✓' if result.accepted else '失败 ✗'}")
    print(f"提取的代码:\n{result.extracted_code}")

    # 计算 reward（示例）
    reward = 1.0 if result.accepted else 0.0
    print(f"\nRL Reward: {reward}")

    return result, reward


def example_submit_safe():
    """
    示例：安全提交（失败不抛异常）
    在批量评测时，使用 submit_safe 可以避免单个失败导致整个批次中断
    """
    print("=" * 60)
    print("示例 5: 安全提交模式")
    print("=" * 60)

    # 故意提交错误代码
    wrong_code = '''
def has_close_elements(numbers, threshold):
    return False  # 错误实现
'''

    result = submit_safe(SubmitRequest(
        dataset='humaneval',
        id='0',
        completion=wrong_code,
        config=TestConfig(language='python')
    ))

    # submit_safe 不会抛异常，而是返回失败结果
    print(f"评测结果: {'通过' if result.accepted else '失败'}")
    print(f"即使代码错误，程序也不会崩溃")

    return result


# =============================================================================
# 第三部分：RL 训练流程示例
# =============================================================================

def simulate_llm_generate(prompt: str) -> str:
    """
    模拟 LLM 生成代码
    在实际 RL 训练中，这里应该调用你的模型
    """
    # 这里用简单的模板返回，实际应该用模型生成
    if "has_close_elements" in prompt:
        return '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
'''
    return "def solution(): pass"


def example_rl_training_loop():
    """
    示例：完整的 RL 训练循环
    展示如何在训练循环中使用 SandboxFusion
    """
    print("=" * 60)
    print("示例 6: RL 训练循环示例")
    print("=" * 60)

    # 1. 获取数据集
    prompts = get_prompts(GetPromptsRequest(
        dataset='humaneval',
        config={'language': 'python'}
    ))

    print(f"加载了 {len(prompts)} 个训练样本")

    # 2. 模拟训练循环（只处理前 5 个样本）
    total_reward = 0
    results = []

    for prompt in prompts[:5]:
        print(f"\n处理问题 {prompt.id}...")

        # 3. 模型生成代码
        completion = simulate_llm_generate(prompt.prompt)

        # 4. 提交评测获取 reward
        result = submit_safe(SubmitRequest(
            dataset='humaneval',
            id=prompt.id,
            completion=completion,
            config=TestConfig(language='python', run_timeout=20)
        ))

        # 5. 计算 reward
        reward = 1.0 if result.accepted else 0.0
        total_reward += reward

        results.append({
            'id': prompt.id,
            'accepted': result.accepted,
            'reward': reward
        })

        print(f"  结果: {'通过' if result.accepted else '失败'}, Reward: {reward}")

    # 6. 统计
    pass_rate = sum(r['reward'] for r in results) / len(results)
    print(f"\n训练批次统计:")
    print(f"  总 Reward: {total_reward}")
    print(f"  通过率: {pass_rate:.2%}")

    return results


# =============================================================================
# 第四部分：批量并发评测
# =============================================================================

def example_batch_evaluation():
    """
    示例：批量并发评测
    在 RL 训练中，通常需要同时评测多个样本以提高效率
    """
    print("=" * 60)
    print("示例 7: 批量并发评测")
    print("=" * 60)

    # 准备多个评测请求
    requests = []

    # 模拟为 5 个问题生成代码
    for i in range(5):
        requests.append(SubmitRequest(
            dataset='humaneval',
            id=str(i),
            completion=f'''
def solution_{i}():
    # 模拟生成的代码
    pass
''',
            config=TestConfig(language='python', run_timeout=20)
        ))

    print(f"准备了 {len(requests)} 个评测请求")

    # 并发执行评测
    print("开始并发评测...")
    results = run_concurrent(
        func=submit_safe,
        args_list=requests,
        max_workers=5  # 并发数
    )

    # 统计结果
    passed = sum(1 for r in results if r.accepted)
    print(f"\n评测完成:")
    print(f"  通过: {passed}/{len(results)}")
    print(f"  通过率: {passed/len(results):.2%}")

    return results


# =============================================================================
# 第五部分：自定义数据评测
# =============================================================================

def example_custom_data_evaluation():
    """
    示例：使用自定义数据进行评测
    当你有自己的测试用例时，可以使用 provided_data 参数
    """
    print("=" * 60)
    print("示例 8: 自定义数据评测")
    print("=" * 60)

    # 自定义测试数据
    custom_problem = {
        'id': 'custom_001',
        'content': '实现一个函数计算两数之和',
        'test': '''
def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(100, 200) == 300
    print("All tests passed!")

test_add()
''',
        'canonical_solution': 'def add(a, b): return a + b'
    }

    # 模型生成的代码
    model_code = '''
def add(a, b):
    return a + b
'''

    # 提交评测
    result = submit(SubmitRequest(
        dataset='autoeval',  # 使用 AutoEval 类型
        id='0',
        completion=model_code,
        config=TestConfig(
            language='python',
            dataset_type='AutoEval',
            provided_data=custom_problem  # 自定义数据
        )
    ))

    print(f"自定义评测结果: {'通过' if result.accepted else '失败'}")
    print(f"测试输出: {result.tests[0].stdout if result.tests else 'N/A'}")

    return result


# =============================================================================
# 第六部分：不同数据集的使用
# =============================================================================

def example_palmath_dataset():
    """
    示例：PAL-Math 数据集（数学推理）
    适合评估模型的数学推理能力
    """
    print("=" * 60)
    print("示例 9: PAL-Math 数据集")
    print("=" * 60)

    # 获取 GSM8k 子集的问题
    prompts = get_prompts(GetPromptsRequest(
        dataset='palmath',
        config={'subset': 'gsm8k'}
    ))

    print(f"GSM8k 数据集共 {len(prompts)} 个问题")
    print(f"\n示例问题:\n{prompts[0].prompt[:500]}...")

    # PAL-Math 需要模型生成 Python 代码来解决数学问题
    # 代码的最后一行 print() 输出答案
    sample_solution = '''
# 问题: Janet 的鸭子每天下 16 个蛋...
eggs_per_day = 16
eggs_for_breakfast = 3
eggs_for_baking = 4
eggs_sold = eggs_per_day - eggs_for_breakfast - eggs_for_baking
price_per_egg = 2
daily_income = eggs_sold * price_per_egg
print(daily_income)  # 输出答案
'''

    print(f"\n示例解答代码:\n{sample_solution}")

    return prompts


def example_codecontests_dataset():
    """
    示例：Code Contests 数据集（竞赛编程）
    基于标准输入输出的评测方式
    """
    print("=" * 60)
    print("示例 10: Code Contests 数据集")
    print("=" * 60)

    prompts = get_prompts(GetPromptsRequest(
        dataset='codecontests',
        config={
            'language': 'cpp',  # 支持多种语言
            'locale': 'en'
        }
    ))

    print(f"Code Contests 数据集共 {len(prompts)} 个问题")

    # 竞赛编程通常使用 stdin/stdout
    sample_cpp_code = '''
#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;

    int sum = 0;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        sum += x;
    }

    cout << sum << endl;
    return 0;
}
'''

    print(f"\n竞赛编程示例（C++）:\n{sample_cpp_code}")

    return prompts


# =============================================================================
# 第七部分：获取详细评测信息
# =============================================================================

def example_detailed_evaluation():
    """
    示例：获取详细的评测信息
    在调试或分析模型错误时非常有用
    """
    print("=" * 60)
    print("示例 11: 详细评测信息")
    print("=" * 60)

    # 提交一个有 bug 的代码
    buggy_code = '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    # Bug: 使用 <= 而不是 <
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:  # 应该是 <
                return True
    return False
'''

    result = submit(SubmitRequest(
        dataset='humaneval',
        id='0',
        completion=buggy_code,
        config=TestConfig(language='python', run_timeout=20)
    ))

    print(f"评测结果: {'通过' if result.accepted else '失败'}")
    print(f"\n提取的代码:\n{result.extracted_code}")

    # 分析测试用例结果
    if result.tests:
        print(f"\n测试用例详情:")
        for i, test in enumerate(result.tests[:3]):  # 显示前 3 个
            print(f"  测试 {i+1}:")
            print(f"    状态: {test.status}")
            if test.stdout:
                print(f"    stdout: {test.stdout[:100]}")
            if test.stderr:
                print(f"    stderr: {test.stderr[:100]}")

    # 额外信息
    if result.extra:
        print(f"\n额外信息: {result.extra}")

    return result


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("   SandboxFusion - RL 训练 Coding Model 数据集使用指南")
    print("=" * 70 + "\n")

    try:
        # 第一部分：获取 Prompts
        print("\n>>> 第一部分：获取数据集 Prompts <<<\n")
        example_get_humaneval_prompts()
        example_get_single_prompt()
        example_get_mbpp_prompts()

        # 第二部分：提交评测
        print("\n>>> 第二部分：提交代码评测 <<<\n")
        example_submit_and_get_reward()
        example_submit_safe()

        # 第三部分：RL 训练流程
        print("\n>>> 第三部分：RL 训练流程 <<<\n")
        example_rl_training_loop()

        # 第四部分：批量评测
        print("\n>>> 第四部分：批量并发评测 <<<\n")
        example_batch_evaluation()

        # 第五部分：自定义数据
        print("\n>>> 第五部分：自定义数据评测 <<<\n")
        example_custom_data_evaluation()

        # 第六部分：其他数据集
        print("\n>>> 第六部分：其他数据集 <<<\n")
        example_palmath_dataset()
        example_codecontests_dataset()

        # 第七部分：详细信息
        print("\n>>> 第七部分：详细评测信息 <<<\n")
        example_detailed_evaluation()

        print("\n" + "=" * 70)
        print("   所有示例执行完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. SandboxFusion 服务正在运行 (make run)")
        print("2. 已安装 sandbox-fusion SDK (pip install sandbox-fusion)")


if __name__ == "__main__":
    main()
