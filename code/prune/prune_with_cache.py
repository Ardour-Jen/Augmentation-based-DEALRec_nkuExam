from surrogate import train
from utils import get_args
from influence_score import get_influence_score
from effort_score import get_effort_score

import torch
import math
import random
import os

if __name__ == '__main__':
    # 加载参数
    args = get_args()
    
    # 创建缓存目录
    cache_dir = f"/root/autodl-tmp/nku/DEALRec-main/cache/{args.data_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 缓存文件路径
    influence_cache = f"{cache_dir}/influence_scores.pt"
    effort_cache = f"{cache_dir}/effort_scores.pt"
    trainer_cache = f"{cache_dir}/trainer.pt"
    
    # 已训练模型路径
    existing_model_path = f"/root/autodl-tmp/nku/DEALRec-main/code/prune/models/{args.data_name}.pth"

    print("=== DEALRec 数据裁剪 (加载已训练模型) ===")
    print(f"数据集: {args.data_name}")
    print(f"模型路径: {existing_model_path}")
    
    # 1. 检查并加载已训练的模型
    if os.path.exists(trainer_cache):
        print(f"✓ 从缓存加载trainer: {trainer_cache}")
        trainer = torch.load(trainer_cache)
    elif os.path.exists(existing_model_path):
        print(f"✓ 发现已训练模型: {existing_model_path}")
        print("🔄 构建trainer并加载模型...")
        
        # 设置do_eval=True来跳过训练，直接加载模型
        args.do_eval = True
        trainer = train(args)  # 这会直接加载existing_model_path
        
        # 保存trainer到缓存
        torch.save(trainer, trainer_cache)
        print(f"✓ 保存trainer到缓存: {trainer_cache}")
    else:
        print(f"❌ 未找到已训练模型: {existing_model_path}")
        print("🔄 开始训练新模型...")
        args.do_eval = False
        trainer = train(args)
        torch.save(trainer, trainer_cache)
        print(f"✓ 训练完成并保存到缓存: {trainer_cache}")
    
    # 2. 计算或加载影响力得分
    if os.path.exists(influence_cache):
        print(f"✓ 从缓存加载影响力得分: {influence_cache}")
        influence = torch.load(influence_cache)
    else:
        print("🔄 计算影响力得分...")
        influence = get_influence_score(args, trainer)
        torch.save(influence, influence_cache)
        print(f"✓ 保存影响力得分到: {influence_cache}")
    
    # 3. 计算或加载努力得分
    if os.path.exists(effort_cache):
        print(f"✓ 从缓存加载努力得分: {effort_cache}")
        effort = torch.load(effort_cache)
    else:
        print("🔄 计算努力得分...")
        effort = get_effort_score(args)
        torch.save(effort, effort_cache)
        print(f"✓ 保存努力得分到: {effort_cache}")
    
    print(f"\n📊 得分统计:")
    print(f"   影响力得分: min={torch.min(influence):.4f}, max={torch.max(influence):.4f}")
    print(f"   努力得分: min={torch.min(effort):.4f}, max={torch.max(effort):.4f}")
    
    # 4. 归一化
    print("🔄 归一化得分...")
    influence_norm = (influence-torch.min(influence))/(torch.max(influence)-torch.min(influence))
    effort_norm = (effort-torch.min(effort))/(torch.max(effort)-torch.min(effort))

    # 5. 计算总体得分
    print(f"🔄 计算总体得分 (lambda={args.lamda})...")
    overall = influence_norm + args.lamda * effort_norm
    scores_sorted, indices = torch.sort(overall, descending=True)

    # 6. 硬剪枝
    n_prune = math.floor(args.hard_prune * len(scores_sorted))
    scores_sorted = scores_sorted[n_prune:]
    indices = indices[n_prune:]
    print(f"✂️ 硬剪枝后剩余样本: {len(scores_sorted)} (剪枝{args.hard_prune*100}%)")

    # 7. 将得分分成k个范围
    s_max = torch.max(scores_sorted)
    s_min = torch.min(scores_sorted)
    print(f"📈 得分范围: {s_min:.4f} ~ {s_max:.4f}")
    interval = (s_max - s_min) / args.k

    s_split = [min(s_min + (interval * _), s_max)for _ in range(1, args.k+1)]

    score_split = [[] for _ in range(args.k)]
    for idxx, s in enumerate(scores_sorted):
        for idx, ref in enumerate(s_split):
            if s.item() <= ref:
                score_split[idx].append({indices[idxx].item():s.item()})
                break
    
    # 8. 覆盖度增强样本选择
    print(f"🎯 开始覆盖度增强选择 (目标: {args.n_fewshot}个样本)...")
    coreset = []
    m = args.n_fewshot
    round_num = 0
    
    while len(score_split):
        round_num += 1
        # select the group with fewest samples
        group = sorted(score_split, key=lambda x:len(x))
        if len(group) > 3:
            print(f"   第{round_num}轮 - 组大小: {len(group[0])}, {len(group[1])}, {len(group[2])}, {len(group[3])}...")
        
        group = [strat for strat in group if len(strat)]
        if len(group) > 3:
            print(f"   过滤空组后: {len(group[0])}, {len(group[1])}, {len(group[2])}, {len(group[3])}...")

        budget = min(len(group[0]), math.floor(m/len(group)))
        print(f"   当前轮预算: {budget}")
        
        # random select and add to the fewshot indices list
        fewest = group[0]
        selected_idx = random.sample([list(_.keys())[0] for _ in fewest], budget)
        coreset.extend(selected_idx)

        # remove the fewest group
        score_split = group[1:]
        m = m - len(selected_idx)
        
    print(f"🎉 样本选择完成! 共选择 {len(coreset)} 个样本")

    # 9. 保存结果
    output_file = f"selected/{args.data_name}_{args.n_fewshot}.pt"
    torch.save(coreset, output_file)
    print(f"💾 结果保存到: {output_file}")
    
    # 10. 保存详细信息
    result_info = {
        'coreset': coreset,
        'influence_scores': influence,
        'effort_scores': effort,
        'overall_scores': overall,
        'args': vars(args)
    }
    detailed_output = f"selected/{args.data_name}_{args.n_fewshot}_detailed.pt"
    torch.save(result_info, detailed_output)
    print(f"📋 详细信息保存到: {detailed_output}")
    
    print("\n" + "="*60)
    print("🚀 DEALRec数据裁剪完成!")
    print(f"✅ 选择样本数: {len(coreset)}")
    print(f"✅ 缓存目录: {cache_dir}")
    print(f"✅ 输出文件: {output_file}")
    print("="*60)