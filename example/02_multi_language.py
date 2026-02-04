"""
SandboxFusion 示例 2: 多语言代码执行
=====================================
本示例展示如何执行多种编程语言的代码。

支持的语言包括:
- 编译型: C++, Go, Java, C#, Rust, Kotlin, Swift, Scala
- 脚本型: Python, Node.js, TypeScript, PHP, Ruby, Lua, R, Perl, Julia
- 测试框架: pytest, Jest, JUnit, go_test
- 特殊: Jupyter, CUDA, Verilog, Lean, SQL, Bash
"""

from sandbox_fusion import run_code, RunCodeRequest, set_endpoint

set_endpoint("http://localhost:8080")


def run_java_example():
    """执行 Java 代码"""
    print("=" * 50)
    print("Java 代码示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
import java.util.ArrayList;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {
        ArrayList<String> fruits = new ArrayList<>();
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");
        fruits.add("Date");

        System.out.println("原始列表: " + fruits);

        Collections.sort(fruits);
        System.out.println("排序后: " + fruits);

        Collections.reverse(fruits);
        System.out.println("反转后: " + fruits);
    }
}
''',
        language='java',
        compile_timeout=30,
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    print()


def run_nodejs_example():
    """执行 Node.js 代码"""
    print("=" * 50)
    print("Node.js 代码示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
// 使用 Promise 和 async/await
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
    console.log("开始执行...");

    const numbers = [1, 2, 3, 4, 5];

    // Map, Filter, Reduce 示例
    const doubled = numbers.map(x => x * 2);
    console.log("Doubled:", doubled);

    const evens = numbers.filter(x => x % 2 === 0);
    console.log("Evens:", evens);

    const sum = numbers.reduce((a, b) => a + b, 0);
    console.log("Sum:", sum);

    // 对象解构
    const person = { name: "Alice", age: 25, city: "Beijing" };
    const { name, age } = person;
    console.log(`${name} is ${age} years old`);

    console.log("执行完成!");
}

main();
''',
        language='nodejs',
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    print()


def run_typescript_example():
    """执行 TypeScript 代码"""
    print("=" * 50)
    print("TypeScript 代码示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
interface Person {
    name: string;
    age: number;
    greet(): string;
}

class Student implements Person {
    constructor(public name: string, public age: number, public grade: string) {}

    greet(): string {
        return `Hello, I'm ${this.name}, ${this.age} years old, in ${this.grade}`;
    }
}

const students: Student[] = [
    new Student("Alice", 20, "Grade A"),
    new Student("Bob", 21, "Grade B"),
    new Student("Charlie", 19, "Grade A")
];

students.forEach(s => console.log(s.greet()));

// 泛型示例
function firstElement<T>(arr: T[]): T | undefined {
    return arr[0];
}

console.log("First student:", firstElement(students)?.name);
''',
        language='typescript',
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    print()


def run_rust_example():
    """执行 Rust 代码"""
    print("=" * 50)
    print("Rust 代码示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
fn main() {
    // Vector 操作
    let mut numbers: Vec<i32> = vec![5, 2, 8, 1, 9];
    println!("Original: {:?}", numbers);

    numbers.sort();
    println!("Sorted: {:?}", numbers);

    // 闭包和迭代器
    let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
    println!("Doubled: {:?}", doubled);

    // 模式匹配
    let result = match numbers.get(2) {
        Some(n) => format!("Third element is {}", n),
        None => String::from("No third element"),
    };
    println!("{}", result);

    // 结构体
    #[derive(Debug)]
    struct Point {
        x: f64,
        y: f64,
    }

    let p = Point { x: 3.0, y: 4.0 };
    let distance = (p.x * p.x + p.y * p.y).sqrt();
    println!("Point {:?} distance from origin: {}", p, distance);
}
''',
        language='rust',
        compile_timeout=60,
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    if response.run_result.stderr:
        print(f"错误/警告: {response.run_result.stderr}")
    print()


def run_bash_example():
    """执行 Bash 脚本"""
    print("=" * 50)
    print("Bash 脚本示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
#!/bin/bash

echo "=== 系统信息 ==="
echo "当前时间: $(date)"
echo "当前目录: $(pwd)"

echo ""
echo "=== 循环示例 ==="
for i in {1..5}; do
    echo "计数: $i"
done

echo ""
echo "=== 数组操作 ==="
fruits=("Apple" "Banana" "Cherry")
for fruit in "${fruits[@]}"; do
    echo "水果: $fruit"
done

echo ""
echo "=== 条件判断 ==="
number=42
if [ $number -gt 40 ]; then
    echo "$number 大于 40"
else
    echo "$number 不大于 40"
fi
''',
        language='bash',
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    print()


def run_ruby_example():
    """执行 Ruby 代码"""
    print("=" * 50)
    print("Ruby 代码示例")
    print("=" * 50)

    response = run_code(RunCodeRequest(
        code='''
# Ruby 代码示例

# 类定义
class Animal
  attr_accessor :name, :age

  def initialize(name, age)
    @name = name
    @age = age
  end

  def speak
    "#{@name} makes a sound"
  end
end

class Dog < Animal
  def speak
    "#{@name} says: Woof!"
  end
end

# 创建对象
dog = Dog.new("Buddy", 3)
puts dog.speak

# 数组操作
numbers = [1, 2, 3, 4, 5]
puts "原数组: #{numbers}"
puts "平方: #{numbers.map { |n| n ** 2 }}"
puts "偶数: #{numbers.select { |n| n.even? }}"
puts "总和: #{numbers.reduce(:+)}"

# Hash 操作
person = { name: "Alice", age: 25, city: "Beijing" }
person.each { |key, value| puts "#{key}: #{value}" }
''',
        language='ruby',
        run_timeout=10
    ))

    print(f"状态: {response.status}")
    print(f"输出:\n{response.run_result.stdout}")
    print()


def main():
    """运行所有多语言示例"""
    print("\n" + "=" * 60)
    print("   SandboxFusion 多语言代码执行示例")
    print("=" * 60 + "\n")

    try:
        run_java_example()
        run_nodejs_example()
        run_typescript_example()
        run_rust_example()
        run_bash_example()
        run_ruby_example()

        print("=" * 60)
        print("   所有多语言示例执行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保 SandboxFusion 服务正在运行，并且已安装相应的语言运行时")


if __name__ == "__main__":
    main()
