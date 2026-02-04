"""
SandboxFusion 示例 4: RL 训练 Reward 集成
==========================================
本示例展示如何将 SandboxFusion 集成到强化学习训练流程中，
用于计算代码生成模型的 reward。

典型的 RL 训练流程:
1. 从数据集采样问题 (prompt)
2. 模型生成代码 (completion)
3. SandboxFusion 评测代码
4. 根据评测结果计算 reward
5. 使用 reward 更新模型

本示例提供:
- RewardCalculator: 封装 reward 计算逻辑
- BatchEvaluator: 批量评测处理
- 与 veRL/OpenRLHF 等框架集成的示例
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import time

from sandbox_fusion import (
    get_prompts,
    get_prompt_by_id,
    submit,
    submit_safe,
    submit_async,
    run_concurrent,
    GetPromptsRequest,
    GetPromptByIdRequest,
    SubmitRequest,
    TestConfig,
    EvalResult,
    set_endpoint
)

set_endpoint("http://localhost:8080")


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class CodeSample:
    """代码样本，包含 prompt 和模型生成的 completion"""
    problem_id: str
    prompt: str
    completion: str
    dataset: str = 'humaneval'


@dataclass
class EvaluationResult:
    """评测结果，包含 reward 信息"""
    problem_id: str
    accepted: bool
    reward: float
    execution_time: float
    error_message: Optional[str] = None
    test_details: Optional[List[Dict]] = None


# =============================================================================
# Reward 计算器
# =============================================================================

class RewardCalculator:
    """
    Reward 计算器
    封装了代码评测和 reward 计算逻辑
    """

    def __init__(
        self,
        dataset: str = 'humaneval',
        language: str = 'python',
        pass_reward: float = 1.0,
        fail_reward: float = 0.0,
        partial_reward: bool = False,
        timeout: float = 20.0
    ):
        """
        初始化 Reward 计算器

        Args:
            dataset: 数据集名称
            language: 编程语言
            pass_reward: 通过时的 reward
            fail_reward: 失败时的 reward
            partial_reward: 是否计算部分通过的 reward
            timeout: 执行超时时间
        """
        self.dataset = dataset
        self.language = language
        self.pass_reward = pass_reward
        self.fail_reward = fail_reward
        self.partial_reward = partial_reward
        self.timeout = timeout

    def calculate_reward(self, result: EvalResult) -> float:
        """
        根据评测结果计算 reward

        Args:
            result: SandboxFusion 的评测结果

        Returns:
            计算得到的 reward 值
        """
        if result.accepted:
            return self.pass_reward

        if self.partial_reward and result.tests:
            # 计算部分通过的 reward
            passed_tests = sum(1 for t in result.tests if t.status == 'Passed')
            total_tests = len(result.tests)
            if total_tests > 0:
                partial = (passed_tests / total_tests) * self.pass_reward
                return max(partial, self.fail_reward)

        return self.fail_reward

    def evaluate_single(self, sample: CodeSample) -> EvaluationResult:
        """
        评测单个代码样本

        Args:
            sample: 代码样本

        Returns:
            评测结果
        """
        start_time = time.time()

        try:
            result = submit_safe(SubmitRequest(
                dataset=sample.dataset,
                id=sample.problem_id,
                completion=sample.completion,
                config=TestConfig(
                    language=self.language,
                    run_timeout=self.timeout
                )
            ))

            reward = self.calculate_reward(result)
            execution_time = time.time() - start_time

            return EvaluationResult(
                problem_id=sample.problem_id,
                accepted=result.accepted,
                reward=reward,
                execution_time=execution_time,
                test_details=[
                    {'status': t.status, 'stdout': t.stdout[:100] if t.stdout else None}
                    for t in (result.tests or [])[:3]
                ]
            )

        except Exception as e:
            return EvaluationResult(
                problem_id=sample.problem_id,
                accepted=False,
                reward=self.fail_reward,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def evaluate_batch(
        self,
        samples: List[CodeSample],
        max_workers: int = 10
    ) -> List[EvaluationResult]:
        """
        批量评测代码样本

        Args:
            samples: 代码样本列表
            max_workers: 最大并发数

        Returns:
            评测结果列表
        """
        # 准备请求
        requests = [
            SubmitRequest(
                dataset=s.dataset,
                id=s.problem_id,
                completion=s.completion,
                config=TestConfig(
                    language=self.language,
                    run_timeout=self.timeout
                )
            )
            for s in samples
        ]

        # 并发评测
        start_time = time.time()
        results = run_concurrent(
            func=submit_safe,
            args_list=requests,
            max_workers=max_workers
        )
        total_time = time.time() - start_time

        # 转换结果
        eval_results = []
        for sample, result in zip(samples, results):
            reward = self.calculate_reward(result)
            eval_results.append(EvaluationResult(
                problem_id=sample.problem_id,
                accepted=result.accepted,
                reward=reward,
                execution_time=total_time / len(samples)  # 平均时间
            ))

        return eval_results


# =============================================================================
# 数据集加载器
# =============================================================================

class DatasetLoader:
    """
    数据集加载器
    用于加载和管理训练数据
    """

    def __init__(self, dataset: str = 'humaneval', language: str = 'python'):
        self.dataset = dataset
        self.language = language
        self._prompts = None

    def load(self) -> List[Dict]:
        """加载数据集"""
        if self._prompts is None:
            prompts = get_prompts(GetPromptsRequest(
                dataset=self.dataset,
                config={'language': self.language}
            ))
            self._prompts = [
                {'id': p.id, 'prompt': p.prompt}
                for p in prompts
            ]
        return self._prompts

    def get_prompt(self, problem_id: str) -> str:
        """获取指定问题的 prompt"""
        prompt = get_prompt_by_id(GetPromptByIdRequest(
            dataset=self.dataset,
            id=problem_id,
            config={'language': self.language}
        ))
        return prompt.prompt

    def sample(self, n: int) -> List[Dict]:
        """随机采样 n 个问题"""
        import random
        prompts = self.load()
        return random.sample(prompts, min(n, len(prompts)))

    def __len__(self):
        return len(self.load())


# =============================================================================
# RL 训练集成示例
# =============================================================================

def example_basic_reward_calculation():
    """示例：基础 reward 计算"""
    print("=" * 60)
    print("示例 1: 基础 Reward 计算")
    print("=" * 60)

    calculator = RewardCalculator(
        dataset='humaneval',
        pass_reward=1.0,
        fail_reward=0.0
    )

    # 正确的代码
    correct_sample = CodeSample(
        problem_id='0',
        prompt='',  # 省略
        completion='''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
'''
    )

    # 错误的代码
    wrong_sample = CodeSample(
        problem_id='0',
        prompt='',
        completion='def has_close_elements(numbers, threshold): return False'
    )

    # 评测
    correct_result = calculator.evaluate_single(correct_sample)
    wrong_result = calculator.evaluate_single(wrong_sample)

    print(f"正确代码 - Accepted: {correct_result.accepted}, Reward: {correct_result.reward}")
    print(f"错误代码 - Accepted: {wrong_result.accepted}, Reward: {wrong_result.reward}")


def example_partial_reward():
    """示例：部分通过的 reward 计算"""
    print("\n" + "=" * 60)
    print("示例 2: 部分通过 Reward（用于更平滑的训练信号）")
    print("=" * 60)

    calculator = RewardCalculator(
        dataset='humaneval',
        pass_reward=1.0,
        fail_reward=0.0,
        partial_reward=True  # 启用部分 reward
    )

    # 部分正确的代码（可能通过部分测试用例）
    partial_sample = CodeSample(
        problem_id='0',
        prompt='',
        completion='''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    # 简单实现，可能只通过部分测试
    if len(numbers) < 2:
        return False
    return abs(numbers[0] - numbers[1]) < threshold
'''
    )

    result = calculator.evaluate_single(partial_sample)
    print(f"部分正确代码 - Accepted: {result.accepted}, Reward: {result.reward:.3f}")
    print(f"测试详情: {result.test_details}")


def example_batch_evaluation():
    """示例：批量评测"""
    print("\n" + "=" * 60)
    print("示例 3: 批量评测（提高训练效率）")
    print("=" * 60)

    calculator = RewardCalculator(dataset='humaneval')
    loader = DatasetLoader(dataset='humaneval')

    # 采样 5 个问题
    sampled = loader.sample(5)

    # 模拟为每个问题生成代码
    samples = [
        CodeSample(
            problem_id=p['id'],
            prompt=p['prompt'],
            completion='def solution(): pass'  # 模拟生成
        )
        for p in sampled
    ]

    # 批量评测
    start = time.time()
    results = calculator.evaluate_batch(samples, max_workers=5)
    elapsed = time.time() - start

    # 统计
    total_reward = sum(r.reward for r in results)
    passed = sum(1 for r in results if r.accepted)

    print(f"评测 {len(samples)} 个样本，耗时 {elapsed:.2f}s")
    print(f"通过: {passed}/{len(samples)}")
    print(f"总 Reward: {total_reward}")
    print(f"平均 Reward: {total_reward/len(samples):.3f}")


def example_rl_training_step():
    """
    示例：完整的 RL 训练步骤
    展示如何在训练循环中使用 SandboxFusion
    """
    print("\n" + "=" * 60)
    print("示例 4: RL 训练步骤")
    print("=" * 60)

    # 初始化
    calculator = RewardCalculator(
        dataset='humaneval',
        pass_reward=1.0,
        fail_reward=-0.1,  # 小的负 reward 惩罚错误
        partial_reward=True
    )
    loader = DatasetLoader(dataset='humaneval')

    # 模拟训练循环
    num_episodes = 3
    batch_size = 4

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        # 1. 采样问题
        batch = loader.sample(batch_size)

        # 2. 模拟模型生成代码（实际应调用你的 LLM）
        samples = []
        for item in batch:
            # 这里应该是: completion = model.generate(item['prompt'])
            completion = f"def solution(): pass  # 问题 {item['id']}"
            samples.append(CodeSample(
                problem_id=item['id'],
                prompt=item['prompt'],
                completion=completion
            ))

        # 3. 获取 rewards
        results = calculator.evaluate_batch(samples, max_workers=batch_size)

        # 4. 计算统计信息
        rewards = [r.reward for r in results]
        mean_reward = sum(rewards) / len(rewards)
        pass_rate = sum(1 for r in results if r.accepted) / len(results)

        print(f"  Mean Reward: {mean_reward:.3f}")
        print(f"  Pass Rate: {pass_rate:.2%}")
        print(f"  Rewards: {rewards}")

        # 5. 这里应该用 rewards 更新模型
        # optimizer.zero_grad()
        # loss = compute_policy_gradient_loss(log_probs, rewards)
        # loss.backward()
        # optimizer.step()


def example_verl_integration():
    """
    示例：与 veRL 框架集成
    展示如何将 SandboxFusion 作为 veRL 的 reward 函数
    """
    print("\n" + "=" * 60)
    print("示例 5: veRL 框架集成示例")
    print("=" * 60)

    # veRL 风格的 reward 函数
    def code_reward_fn(
        prompts: List[str],
        completions: List[str],
        problem_ids: List[str],
        **kwargs
    ) -> List[float]:
        """
        veRL 兼容的 reward 函数

        Args:
            prompts: 输入 prompts 列表
            completions: 模型生成的 completions 列表
            problem_ids: 问题 ID 列表

        Returns:
            reward 列表
        """
        calculator = RewardCalculator(
            dataset=kwargs.get('dataset', 'humaneval'),
            pass_reward=kwargs.get('pass_reward', 1.0),
            fail_reward=kwargs.get('fail_reward', 0.0)
        )

        samples = [
            CodeSample(
                problem_id=pid,
                prompt=prompt,
                completion=completion
            )
            for pid, prompt, completion in zip(problem_ids, prompts, completions)
        ]

        results = calculator.evaluate_batch(samples)
        return [r.reward for r in results]

    # 模拟调用
    rewards = code_reward_fn(
        prompts=['prompt1', 'prompt2'],
        completions=['def f(): pass', 'def g(): return 1'],
        problem_ids=['0', '1'],
        dataset='humaneval'
    )

    print(f"veRL reward 函数返回: {rewards}")
    print("\n在 veRL 中使用:")
    print("""
from verl import ...

# 配置 reward 函数
config.reward_fn = code_reward_fn
config.reward_fn_kwargs = {
    'dataset': 'humaneval',
    'pass_reward': 1.0,
    'fail_reward': 0.0
}
""")


# =============================================================================
# 异步评测示例
# =============================================================================

async def example_async_evaluation():
    """
    示例：异步评测
    在需要高并发时使用
    """
    print("\n" + "=" * 60)
    print("示例 6: 异步评测")
    print("=" * 60)

    async def evaluate_one(problem_id: str, completion: str) -> float:
        """异步评测单个样本"""
        result = await submit_async(SubmitRequest(
            dataset='humaneval',
            id=problem_id,
            completion=completion,
            config=TestConfig(language='python')
        ))
        return 1.0 if result.accepted else 0.0

    # 并发评测多个样本
    tasks = [
        evaluate_one('0', 'def has_close_elements(n, t): return False'),
        evaluate_one('1', 'def separate_paren_groups(s): return []'),
        evaluate_one('2', 'def truncate_number(n): return n % 1'),
    ]

    rewards = await asyncio.gather(*tasks)
    print(f"异步评测结果: {rewards}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("   SandboxFusion - RL 训练 Reward 集成指南")
    print("=" * 70)

    try:
        example_basic_reward_calculation()
        example_partial_reward()
        example_batch_evaluation()
        example_rl_training_step()
        example_verl_integration()

        # 异步示例
        asyncio.run(example_async_evaluation())

        print("\n" + "=" * 70)
        print("   所有示例执行完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保 SandboxFusion 服务正在运行 (make run)")


if __name__ == "__main__":
    main()
