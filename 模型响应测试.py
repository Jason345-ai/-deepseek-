import pandas as pd
from ollama import generate
import time
import os

# 配置参数
MODEL_NAME = "deepseek-r1:1.5b"
TEST_CASES = [
    ("北京今日新增确诊病例5例，卫健委已通报", "真新闻"),
    ("外星人已控制美国政府高层官员", "假新闻"),
    ("4chan网友发现波士顿爆炸案嫌疑人身份", "假新闻"),
    ("教育部宣布明年起取消英语高考", "假新闻"),
    ("科学家证实每天喝咖啡可延长寿命10年", "假新闻"),
    ("国务院发布2024年节假日安排通知", "真新闻")
]

def generate_prompt(text):
    """生成优化的prompt"""
    return f"""请严格按以下要求判断新闻真实性：
1. 只回答"真新闻"或"假新闻"，不要解释
2. 判断标准：
   - 有官方通报/权威信源 → 真新闻
   - 含未经证实/明显荒谬内容 → 假新闻

新闻内容：{text}
判断结果："""

def parse_response(response):
    """增强版响应解析"""
    response = str(response).strip().lower()
    if not response:
        return None
    
    # 检查常见真新闻关键词
    true_keywords = ['真', 'real', 'true', 'correct', 'accurate']
    if any(kw in response for kw in true_keywords):
        return '真新闻'
    
    # 检查常见假新闻关键词
    false_keywords = ['假', 'fake', 'false', 'misinfo', 'rumor']
    if any(kw in response for kw in false_keywords):
        return '假新闻'
    
    # 检查第一个词
    first_word = response.split()[0] if response else ""
    if first_word in ['真', 'true']:
        return '真新闻'
    elif first_word in ['假', 'false']:
        return '假新闻'
    
    return None

def test_single_case(text, expected):
    """测试单个案例"""
    print(f"\n{'='*50}")
    print(f"测试文本: {text}")
    
    try:
        prompt = generate_prompt(text)
        print(f"\n生成的Prompt:\n{prompt}")
        
        start_time = time.time()
        response = generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.1, "num_ctx": 2048},
            stream=False
        )
        latency = time.time() - start_time
        
        raw_response = response['response']
        parsed = parse_response(raw_response)
        
        print(f"\n模型原始响应: {raw_response}")
        print(f"解析结果: {parsed}")
        print(f"预期结果: {expected}")
        print(f"耗时: {latency:.2f}s")
        
        return parsed == expected
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

def main():
    print("="*50)
    print("开始测试DeepSeek-R1模型响应能力")
    print(f"测试案例数: {len(TEST_CASES)}")
    print("="*50)
    
    success_count = 0
    for i, (text, expected) in enumerate(TEST_CASES, 1):
        print(f"\n▶ 测试案例 {i}/{len(TEST_CASES)}")
        if test_single_case(text, expected):
            success_count += 1
            print("✅ 测试通过")
        else:
            print("❌ 测试失败")
        time.sleep(1)  # 避免请求过载
    
    print("\n" + "="*50)
    print(f"测试完成 | 通过率: {success_count}/{len(TEST_CASES)} ({success_count/len(TEST_CASES):.0%})")
    print("="*50)

if __name__ == "__main__":
    main()