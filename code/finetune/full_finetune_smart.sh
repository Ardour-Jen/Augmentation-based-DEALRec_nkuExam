#!/bin/bash

dataset=$1
fewshot=1024

echo "开始智能微调流程：${dataset} 数据集 (小规模测试版本)"
echo "包含：检查已存在模型 -> 微调/恢复 -> 小规模推理 -> 小规模评估"

for seed in 2023
do
    for sample in 1024
    do  
        model_dir="./model/${dataset}/${seed}_${sample}"
        result_file="./results/${dataset}/${seed}_${sample}_small.json"
        
        echo "==== 检查模型状态 ===="
        echo "模型目录: $model_dir"
        echo "结果文件: $result_file"
        
        # 检查模型是否已存在
        if [ -f "$model_dir/adapter_model.bin" ]; then
            echo "✅ 发现已训练的模型！"
            echo "📍 位置: $model_dir"
            
            # 检查推理结果是否已存在
            if [ -f "$result_file" ]; then
                echo "✅ 推理结果也已存在，跳过整个流程"
                echo "📄 结果文件: $result_file"
                echo "如需重新训练，请删除模型目录: rm -rf $model_dir"
                continue
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
            echo "==== 开始微调 ===="
            echo "随机种子: $seed, 样本数: $sample"
            
            # 创建日志目录
            log_dir="./logs/${dataset}/${seed}_${sample}"
            mkdir -p $log_dir
            
            # 微调训练 - 基于原始作者的完整配置
            # 原始: batch_size=128, micro_batch_size=16, num_epochs=50
            # 4090适配版本: 降低batch_size但保持训练充分性
            CUDA_VISIBLE_DEVICES=0 python finetune.py \
                --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75 \
                --train_data_path ./data/${dataset}/train/train-${sample}.json \
                --val_data_path ./data/${dataset}/valid/valid-${sample}.json \
                --output_dir $model_dir \
                --batch_size 128 \
                --micro_batch_size 16 \
                --num_epochs 30 \
                --learning_rate 1e-4 \
                --cutoff_len 512 \
                --lora_r 8 \
                --lora_alpha 16 \
                --lora_dropout 0.05 \
                --lora_target_modules '[q_proj,v_proj,k_proj,o_proj]' \
                --train_on_inputs \
                --group_by_length \
                --seed $seed \
                --sample $sample \
                --log_file $log_dir/training_detailed.csv \
                2>&1 | tee $log_dir/training_console.log
            
            echo "==== 微调完成 ===="
        else
            echo "==== 跳过训练，使用已存在模型 ===="
        fi
        
        echo "==== 开始小规模推理 ===="
        
        # 小规模推理 (使用inference_small.py)
        CUDA_VISIBLE_DEVICES=0 python inference_small.py \
            --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75 \
            --lora_weights $model_dir \
            --result_json_data $result_file \
            --dataset ${dataset}

        echo "==== 推理完成，开始小规模评估 ===="
        
        # 小规模评估 (只传递支持的参数)
        echo "当前工作目录: $(pwd)"
        CUDA_VISIBLE_DEVICES=0 python evaluate_small.py \
            --result_file $result_file \
            --dataset ${dataset} \
            --batch_size 4
        
        echo "==== 完整流程完成！===="
        echo "模型保存位置: $model_dir"
        echo "结果保存位置: $result_file"
        echo "=========================================="
    done
done

echo "🎉 所有任务完成！(小规模测试版本)" 