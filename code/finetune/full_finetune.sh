#!/bin/bash

dataset=$1
fewshot=1024

echo "开始完整的微调流程：${dataset} 数据集"
echo "包含：微调 -> 推理 -> 评估"

for seed in 2023
do
    for sample in 1024
    do  
        echo "==== 开始微调 ===="
        echo "随机种子: $seed, 样本数: $sample"
        
        # 检查是否存在已训练的模型
        model_path="./model/${dataset}/${seed}_${sample}"
        resume_flag=""
        
        if [ -f "$model_path/adapter_model.bin" ]; then
            echo "🔍 发现已存在的模型: $model_path"
            echo "📝 将从检查点恢复训练..."
            resume_flag="--resume_from_checkpoint $model_path"
        else
            echo "🆕 开始全新训练..."
        fi
        
        # 少样本微调 (优化显存利用)
        CUDA_VISIBLE_DEVICES=0 python finetune.py \
            --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75 \
            --train_data_path ./data/${dataset}/train/train-${sample}.json \
            --val_data_path ./data/${dataset}/valid/valid-${sample}.json \
            --output_dir ./model/${dataset}/${seed}_${sample} \
            --batch_size 64 \
            --micro_batch_size 8 \
            --num_epochs 5 \
            --learning_rate 1e-4 \
            --cutoff_len 512 \
            --lora_r 16 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --lora_target_modules '[q_proj,v_proj,k_proj,o_proj]' \
            --train_on_inputs \
            --group_by_length \
            --seed $seed \
            --sample $sample \
            $resume_flag
        
        echo "==== 微调完成，开始推理 ===="
        
        # 生成推理结果
        CUDA_VISIBLE_DEVICES=0 python inference_ddp.py \
            --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75 \
            --lora_weights ./model/${dataset}/${seed}_${sample} \
            --result_json_data ./results/${dataset}/${seed}_${sample}.json \
            --dataset ${dataset}

        echo "==== 推理完成，开始评估 ===="
        
        # 评估结果
        cd data/
        PWD=$(pwd)
        echo "当前工作目录: $PWD"

        gpu_id=0
        res_file=${seed}_${sample}
        sh evaluate.sh ${res_file} ${gpu_id}
        cd ../
        
        echo "==== 完整流程完成！===="
        echo "模型保存位置: ./model/${dataset}/${seed}_${sample}"
        echo "结果保存位置: ./results/${dataset}/${seed}_${sample}.json"
    done
done 