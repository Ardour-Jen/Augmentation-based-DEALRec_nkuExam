#!/bin/bash

dataset=$1
fewshot=1024

echo "开始 ${dataset} 数据集的少样本微调..."

for seed in 2023
do
    for sample in 1024
    do  
        echo "随机种子: $seed, 样本数: $sample"
        
        # 仅进行少样本微调
        CUDA_VISIBLE_DEVICES=0 python finetune.py \
            --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75 \
            --train_data_path ./data/${dataset}/train/train-${sample}.json \
            --val_data_path ./data/${dataset}/valid/valid-${sample}.json \
            --output_dir ./model/${dataset}/${seed}_${sample} \
            --batch_size 8 \
            --micro_batch_size 1 \
            --num_epochs 3 \
            --learning_rate 1e-4 \
            --cutoff_len 128 \
            --lora_r 4 \
            --lora_alpha 8 \
            --lora_dropout 0.05 \
            --lora_target_modules '[q_proj,v_proj]' \
            --train_on_inputs \
            --group_by_length \
            --seed $seed \
            --sample $sample
        
        echo "微调完成! 模型保存在: ./model/${dataset}/${seed}_${sample}"
    done
done 