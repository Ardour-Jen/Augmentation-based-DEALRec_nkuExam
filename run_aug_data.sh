#!/bin/bash

# 数据增强脚本 - 从1024样本生成2048样本
echo "🔧 开始数据增强：从1024样本生成2048样本"

cd code/finetune/data

# 设置模型路径
MODEL_PATH="/root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75"

echo "🔧 生成增强训练数据..."
python gen_fewshot_aug.py \
    --base_model="$MODEL_PATH" \
    --input_dir="../../../data/" \
    --output_dir="train" \
    --n_sample=1024 \
    --dataset="games" \
    --cutoff_len=512 \
    --seed=2023 \
    --augment=True

echo "🔧 生成增强验证数据..."
python gen_fewshot_aug.py \
    --base_model="$MODEL_PATH" \
    --input_dir="../../../data/" \
    --output_dir="valid" \
    --n_sample=1024 \
    --dataset="games" \
    --cutoff_len=512 \
    --seed=2023 \
    --augment=True

echo "🔧 数据增强完成！"
echo "🔧 生成的文件："
echo "   - games/train/train-2048.json (训练数据)"
echo "   - games/valid/valid-2048.json (验证数据)"

cd ../../.. 