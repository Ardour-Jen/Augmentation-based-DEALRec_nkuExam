#!/bin/bash

# 增强数据集微调脚本 - 1932样本训练
echo "🔧 开始增强数据集完整流程 (1932样本)"
echo "包含：微调训练 -> 小规模推理 -> 小规模评估"

# 设置路径
MODEL_PATH="/root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75"
TRAIN_DATA_PATH="code/finetune/data/games/train/train-1932.json"
VAL_DATA_PATH="code/finetune/data/games/valid/valid-365.json"
OUTPUT_DIR="code/finetune/model/games/2023_1932_aug"
RESULT_FILE="code/finetune/results/games/2023_1932_aug_small.json"

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "code/finetune/results/games"
LOG_DIR="code/finetune/logs/games/2023_1932_aug"
mkdir -p "$LOG_DIR"

echo "==== 检查模型状态 ===="
echo "模型目录: $OUTPUT_DIR"
echo "结果文件: $RESULT_FILE"

# 检查模型是否已存在
if [ -f "$OUTPUT_DIR/adapter_model.bin" ]; then
    echo "✅ 发现已训练的模型！"
    echo "📍 位置: $OUTPUT_DIR"
    
    # 检查推理结果是否已存在
    if [ -f "$RESULT_FILE" ]; then
        echo "✅ 推理结果也已存在，跳过整个流程"
        echo "📄 结果文件: $RESULT_FILE"
        echo "如需重新训练，请删除模型目录: rm -rf $OUTPUT_DIR"
        exit 0
    else
        echo "❌ 推理结果不存在，直接进行推理"
        skip_training=true
    fi
else
    echo "❌ 未发现已训练模型，开始从头训练"
    skip_training=false
fi

# 根据检查结果决定是否训练
if [ "$skip_training" = false ]; then
    echo "==== 开始微调训练 ===="
    echo "🔧 训练参数（与full_finetune_smart.sh保持一致）："
    echo "   - 基础模型: LLaMA-7B"
    echo "   - 训练数据: $TRAIN_DATA_PATH"
    echo "   - 验证数据: $VAL_DATA_PATH"
    echo "   - 训练样本: 1932个"
    echo "   - 验证样本: 365个"
    echo "   - 批次大小: 128"
    echo "   - 微批次大小: 16"
    echo "   - 训练轮数: 30"
    echo "   - 学习率: 1e-4"
    echo "   - LoRA rank: 8"
    echo "   - LoRA alpha: 16"

    # 开始训练 - 使用与full_finetune_smart.sh完全相同的参数
    CUDA_VISIBLE_DEVICES=0 python code/finetune/finetune.py \
        --base_model="$MODEL_PATH" \
        --train_data_path="$TRAIN_DATA_PATH" \
        --val_data_path="$VAL_DATA_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --batch_size=128 \
        --micro_batch_size=16 \
        --num_epochs=30 \
        --learning_rate=1e-4 \
        --cutoff_len=512 \
        --lora_r=8 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
        --train_on_inputs \
        --group_by_length \
        --seed=2023 \
        --sample=-1 \
        --log_file="$LOG_DIR/training_detailed.csv" \
        2>&1 | tee "$LOG_DIR/training_console.log"

    echo "==== 微调训练完成 ===="
else
    echo "==== 跳过训练，使用已存在模型 ===="
fi

echo "==== 开始小规模推理 ===="

# 小规模推理 (使用inference_small.py)
CUDA_VISIBLE_DEVICES=0 python code/finetune/inference_small.py \
    --base_model="$MODEL_PATH" \
    --lora_weights="$OUTPUT_DIR" \
    --result_json_data="$RESULT_FILE" \
    --dataset="games" \
    --batch_size=4 \
    --beam_size=2 \
    --max_new_tokens=96 \
    --use_small_test=True

echo "==== 推理完成，开始小规模评估 ===="

# 小规模评估
echo "当前工作目录: $(pwd)"
CUDA_VISIBLE_DEVICES=0 python code/finetune/evaluate_small.py \
    --result_file="$RESULT_FILE" \
    --dataset="games" \
    --batch_size=4

echo "==== 完整流程完成！===="
echo "🔧 模型保存位置: $OUTPUT_DIR"
echo "🔧 结果保存位置: $RESULT_FILE"
echo "🔧 日志保存位置: $LOG_DIR"
echo "=========================================="

echo "🎉 增强数据集实验完成！(1932样本)" 