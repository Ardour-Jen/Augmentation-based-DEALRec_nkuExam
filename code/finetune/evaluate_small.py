#!/usr/bin/env python3
"""
小测试集评估脚本
基于语义相似度的推荐系统评估
"""

import os
import json
import numpy as np
import torch
import math
from tqdm import tqdm
import argparse

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/root/autodl-tmp/nku/DEALRec-main/models_cache'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/nku/DEALRec-main/models_cache'

from transformers import LlamaForCausalLM, LlamaTokenizer

def setup_arguments():
    parser = argparse.ArgumentParser(description="小测试集评估")
    parser.add_argument("--result_file", type=str, 
                       default="./results/games/2023_1024_small.json",
                       help="推理结果文件路径")
    parser.add_argument("--dataset", type=str, default="games", 
                       help="数据集名称")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批处理大小")
    return parser.parse_args()

class RecommendationEvaluator:
    def __init__(self, dataset="games", batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 初始化评估器...")
        print(f"📊 数据集: {dataset}")
        print(f"🖥️  设备: {self.device}")
        
        # 加载模型和tokenizer
        self._load_model()
        
        # 加载物品映射
        self._load_item_mappings()
        
        # 计算物品嵌入
        self._compute_item_embeddings()
    
    def _load_model(self):
        """加载LLaMA模型和tokenizer"""
        print("🤖 加载LLaMA模型...")
        base_model = "/root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75"
        
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # 配置模型
        self.model.half()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.tokenizer.padding_side = "left"
        self.model.eval()
        
        print("✅ 模型加载完成")
    
    def _load_item_mappings(self):
        """加载物品映射关系"""
        print("📋 加载物品映射...")
        try:
            # 尝试加载原始映射文件
            mapping_file = f'../../data/{self.dataset}/title_maps.npy'
            if os.path.exists(mapping_file):
                movies = np.load(mapping_file, allow_pickle=True).item()
                self.item_names = list(movies['seqid2title'].values())
                print(f"✅ 从 {mapping_file} 加载了 {len(self.item_names)} 个物品")
            else:
                # 如果映射文件不存在，从测试数据中提取
                print("⚠️ 映射文件不存在，从测试数据中提取物品名称...")
                self._extract_items_from_test_data()
        except Exception as e:
            print(f"❌ 加载映射失败: {e}")
            print("📝 从测试数据中提取物品名称...")
            self._extract_items_from_test_data()
        
        # 创建物品字典
        self.item_dict = {name: idx for idx, name in enumerate(self.item_names)}
        print(f"📊 物品映射创建完成，共 {len(self.item_names)} 个物品")
    
    def _extract_items_from_test_data(self):
        """从测试数据中提取所有物品名称"""
        # 从小测试集中提取物品
        test_file = f'data/{self.dataset}/test/test_small.json'
        if not os.path.exists(test_file):
            test_file = f'data/{self.dataset}/test/test.json'
        
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # 提取所有unique的物品名称
        all_items = set()
        for item in test_data:
            if isinstance(item['output'], list):
                for out in item['output']:
                    if isinstance(out, str):
                        all_items.add(out.strip('"'))
            elif isinstance(item['output'], str):
                # 尝试解析字符串格式的列表
                try:
                    import ast
                    parsed = ast.literal_eval(item['output'])
                    if isinstance(parsed, list):
                        for out in parsed:
                            all_items.add(str(out).strip('"'))
                except:
                    all_items.add(item['output'].strip('"'))
        
        self.item_names = list(all_items)
        print(f"📝 从测试数据中提取了 {len(self.item_names)} 个unique物品")
    
    def _batch_process(self, items, batch_size):
        """批处理助手函数"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def _compute_item_embeddings(self):
        """计算所有物品的嵌入表示"""
        print("🧮 计算物品嵌入...")
        item_embeddings = []
        
        with torch.no_grad():
            for batch_items in tqdm(
                self._batch_process(self.item_names, self.batch_size),
                desc="计算物品嵌入",
                total=(len(self.item_names) + self.batch_size - 1) // self.batch_size
            ):
                inputs = self.tokenizer(
                    batch_items, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                # 使用最后一层的最后一个token的隐藏状态
                hidden_states = outputs.hidden_states[-1][:, -1, :]
                item_embeddings.append(hidden_states.detach().cpu())
        
        self.item_embeddings = torch.cat(item_embeddings, dim=0).to(self.device)
        print(f"✅ 物品嵌入计算完成: {self.item_embeddings.shape}")
    
    def _compute_prediction_embeddings(self, predictions):
        """计算预测文本的嵌入表示"""
        print("🔮 计算预测嵌入...")
        pred_embeddings = []
        
        with torch.no_grad():
            for batch_preds in tqdm(
                self._batch_process(predictions, self.batch_size),
                desc="计算预测嵌入",
                total=(len(predictions) + self.batch_size - 1) // self.batch_size
            ):
                inputs = self.tokenizer(
                    batch_preds,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1][:, -1, :]
                pred_embeddings.append(hidden_states.detach().cpu())
        
        return torch.cat(pred_embeddings, dim=0).to(self.device)
    
    def _calculate_ndcg(self, ranks, ground_truth, topk):
        """计算NDCG@K"""
        ndcg_sum = 0
        
        for i, gt_items in enumerate(ground_truth):
            dcg = 0
            idcg = 0
            
            # 计算DCG
            for j, item in enumerate(gt_items):
                if item in self.item_dict:
                    item_id = self.item_dict[item]
                    rank = ranks[i][item_id].item()
                    if rank < topk:
                        dcg += 1 / math.log2(rank + 2)
                
                # 计算IDCG (理想排序)
                if j < topk:
                    idcg += 1 / math.log2(j + 2)
            
            # 避免除零
            if idcg > 0:
                ndcg_sum += dcg / idcg
        
        return ndcg_sum / len(ground_truth)
    
    def _calculate_recall(self, ranks, ground_truth, topk):
        """计算Recall@K"""
        recall_sum = 0
        
        for i, gt_items in enumerate(ground_truth):
            hits = 0
            for item in gt_items:
                if item in self.item_dict:
                    item_id = self.item_dict[item]
                    rank = ranks[i][item_id].item()
                    if rank < topk:
                        hits += 1
            
            if len(gt_items) > 0:
                recall_sum += hits / len(gt_items)
        
        return recall_sum / len(ground_truth)
    
    def evaluate(self, result_file):
        """执行评估"""
        print(f"📊 开始评估: {result_file}")
        
        # 加载推理结果
        with open(result_file, 'r') as f:
            test_data = json.load(f)
        
        print(f"📝 加载了 {len(test_data)} 个测试样本")
        
        # 提取预测文本和真实标签
        predictions = []
        ground_truth = []
        
        for item in test_data:
            pred = item['predict'].strip('"')
            predictions.append(pred)
            
            # 处理真实标签
            gt = item['output']
            if isinstance(gt, str):
                try:
                    import ast
                    gt = ast.literal_eval(gt)
                except:
                    gt = [gt]
            if isinstance(gt, list):
                gt = [str(x).strip('"') for x in gt]
            else:
                gt = [str(gt).strip('"')]
            
            ground_truth.append(gt)
        
        # 计算预测嵌入
        pred_embeddings = self._compute_prediction_embeddings(predictions)
        
        # 计算距离和排名
        print("📐 计算相似度距离...")
        distances = torch.cdist(pred_embeddings, self.item_embeddings, p=2)
        ranks = distances.argsort(dim=-1).argsort(dim=-1)
        
        # 在不同K值下评估
        topk_list = [5, 10, 20, 50]
        results = {}
        
        print("📈 计算评估指标...")
        for topk in topk_list:
            ndcg = self._calculate_ndcg(ranks, ground_truth, topk)
            recall = self._calculate_recall(ranks, ground_truth, topk)
            
            results[f'NDCG@{topk}'] = round(ndcg, 4)
            results[f'Recall@{topk}'] = round(recall, 4)
        
        return results
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*60)
        print("📊 评估结果")
        print("="*60)
        
        # 打印表格
        print(f"{'指标':<12} {'@5':<8} {'@10':<8} {'@20':<8} {'@50':<8}")
        print("-"*50)
        
        ndcg_values = [results[f'NDCG@{k}'] for k in [5, 10, 20, 50]]
        recall_values = [results[f'Recall@{k}'] for k in [5, 10, 20, 50]]
        
        print(f"{'NDCG':<12} {ndcg_values[0]:<8} {ndcg_values[1]:<8} {ndcg_values[2]:<8} {ndcg_values[3]:<8}")
        print(f"{'Recall':<12} {recall_values[0]:<8} {recall_values[1]:<8} {recall_values[2]:<8} {recall_values[3]:<8}")
        
        print("\n💡 指标解释:")
        print("  NDCG: 考虑排序位置的准确率 (越高越好)")
        print("  Recall: 召回率，找到的相关物品比例 (越高越好)")

def main():
    args = setup_arguments()
    
    # 检查结果文件是否存在
    if not os.path.exists(args.result_file):
        print(f"❌ 结果文件不存在: {args.result_file}")
        print("请先运行 inference_small.py 生成推理结果")
        return
    
    # 初始化评估器
    evaluator = RecommendationEvaluator(
        dataset=args.dataset,
        batch_size=args.batch_size
    )
    
    # 执行评估
    results = evaluator.evaluate(args.result_file)
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存结果
    output_file = args.result_file.replace('.json', '_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 评估结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 