#!/usr/bin/env python3
import json

def create_test_subset():
    # 读取原始测试数据
    print("📖 读取原始测试数据...")
    with open('data/games/test/test.json', 'r') as f:
        data = json.load(f)
    
    print(f"✅ 原始测试数据量: {len(data)}")
    
    # 取前100个数据
    test_subset = data[:100]
    
    # 保存为新的测试文件
    print("💾 创建小测试集...")
    with open('data/games/test/test_small.json', 'w') as f:
        json.dump(test_subset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已创建小测试集: {len(test_subset)} 个样本")
    print("📍 保存位置: data/games/test/test_small.json")
    
    # 显示前2个样本的结构
    print("\n📋 前2个样本结构:")
    for i, item in enumerate(test_subset[:2]):
        print(f"{i+1}. instruction: {item['instruction'][:80]}...")
        print(f"   input: {item['input'][:80]}...")
        print(f"   output: {item['output'][:80]}...")
        print()
    
    return test_subset

if __name__ == "__main__":
    create_test_subset()
    print("🎉 测试子集创建完成!") 