#!/usr/bin/env python3
"""
推理方法对比测试脚本
比较原始后处理和当前后处理的差异
"""

import json
import os
from difflib import SequenceMatcher

def original_postprocess(text):
    """原始inference_ddp.py的后处理方法"""
    return text.split('Response:\n')[-1] if 'Response:\n' in text else text

def current_postprocess(text):
    """当前inference_small.py的后处理方法"""
    if 'Response:\n' in text:
        response = text.split('Response:\n')[-1].strip()
    elif 'Response:' in text:
        response = text.split('Response:')[-1].strip()
    elif '### Response:' in text:
        response = text.split('### Response:')[-1].strip()
    else:
        lines = text.strip().split('\n')
        response_lines = []
        skip_prompt = False
        for line in lines:
            if 'Below is an instruction' in line or '### Instruction:' in line or '### Input:' in line:
                skip_prompt = True
                continue
            if '### Response:' in line:
                skip_prompt = False
                continue
            if not skip_prompt and line.strip():
                response_lines.append(line.strip())
        
        response = ' '.join(response_lines[-3:]) if response_lines else text[-100:]
    
    response = response.replace('###', '').replace('Instruction:', '').replace('Input:', '').strip()
    return response

def similarity_score(text1, text2):
    """计算两个文本的相似度"""
    return SequenceMatcher(None, text1, text2).ratio()

def analyze_postprocessing_differences():
    """分析后处理方法的差异"""
    
    # 测试样例：模拟不同格式的模型输出
    test_cases = [
        # 标准格式
        """Below is an instruction that describes a task.

### Instruction:
Recommend games for user.

### Response:
The Witcher 3, Cyberpunk 2077, Red Dead Redemption 2""",
        
        # 带换行的Response格式
        """Below is an instruction that describes a task.

### Instruction:
Recommend games for user.

### Response:
The Witcher 3
Cyberpunk 2077
Red Dead Redemption 2""",
        
        # 不标准格式
        """Some prompt text here
Response: The Witcher 3, Cyberpunk 2077""",
        
        # 复杂格式
        """### Instruction:
Recommend games
### Input:
User likes RPG
### Response:
Final Fantasy, Dragon Age
Some extra text here""",
        
        # 边界情况
        """No clear response format
Just some game recommendations: Skyrim, Fallout""",
    ]
    
    print("🔍 推理后处理方法对比分析")
    print("=" * 80)
    
    total_similarity = 0
    significant_differences = 0
    
    for i, test_case in enumerate(test_cases, 1):
        original_result = original_postprocess(test_case)
        current_result = current_postprocess(test_case)
        
        similarity = similarity_score(original_result, current_result)
        total_similarity += similarity
        
        print(f"\n📝 测试案例 {i}:")
        print(f"原始方法结果: {repr(original_result[:100])}...")
        print(f"当前方法结果: {repr(current_result[:100])}...")
        print(f"相似度: {similarity:.3f}")
        
        if similarity < 0.8:
            significant_differences += 1
            print("⚠️  显著差异！")
        elif similarity < 0.95:
            print("🟡 中等差异")
        else:
            print("✅ 基本一致")
    
    avg_similarity = total_similarity / len(test_cases)
    
    print("\n" + "=" * 80)
    print("📊 总体分析结果:")
    print(f"平均相似度: {avg_similarity:.3f}")
    print(f"显著差异案例: {significant_differences}/{len(test_cases)}")
    
    if avg_similarity < 0.8:
        print("🔴 高风险：后处理差异可能显著影响评估结果")
    elif avg_similarity < 0.9:
        print("🟡 中等风险：后处理差异可能影响部分评估结果")
    else:
        print("✅ 低风险：后处理差异影响较小")
    
    return avg_similarity, significant_differences

def check_existing_results():
    """检查现有推理结果的格式分布"""
    result_file = "./results/games/2023_1024_small.json"
    
    if not os.path.exists(result_file):
        print(f"⚠️ 结果文件不存在: {result_file}")
        return
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n🔍 分析现有推理结果: {result_file}")
    print(f"样本数量: {len(data)}")
    
    # 分析预测文本的格式特征
    format_stats = {
        'has_response_newline': 0,
        'has_response_colon': 0,
        'has_hash_response': 0,
        'avg_length': 0,
        'empty_predictions': 0
    }
    
    total_length = 0
    sample_predictions = []
    
    for item in data[:10]:  # 只看前10个样本
        pred = item.get('predict', '')
        sample_predictions.append(pred)
        
        if not pred.strip():
            format_stats['empty_predictions'] += 1
        
        total_length += len(pred)
        
        if 'Response:\n' in pred:
            format_stats['has_response_newline'] += 1
        if 'Response:' in pred:
            format_stats['has_response_colon'] += 1
        if '### Response:' in pred:
            format_stats['has_hash_response'] += 1
    
    format_stats['avg_length'] = total_length / len(data) if data else 0
    
    print("📈 格式统计:")
    for key, value in format_stats.items():
        if key == 'avg_length':
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n📋 前3个预测样本:")
    for i, pred in enumerate(sample_predictions[:3], 1):
        print(f"样本 {i}: {repr(pred[:100])}...")

if __name__ == "__main__":
    # 分析后处理方法差异
    avg_sim, sig_diff = analyze_postprocessing_differences()
    
    # 检查现有结果
    check_existing_results()
    
    print(f"\n💡 建议:")
    if avg_sim < 0.9 or sig_diff > 0:
        print("🔧 建议使用原始后处理方法以确保结果一致性")
        print("📋 可以运行以下命令重新生成推理结果:")
        print("   python inference_ddp.py --batch_size=1 --beam_size=2")
    else:
        print("✅ 当前后处理方法影响较小，可以继续使用") 