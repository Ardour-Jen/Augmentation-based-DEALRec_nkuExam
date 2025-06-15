accelerate config

dataset=$1
fewshot=1024

# generate data for LLM-based recommender models (BIGRec)
cd data/
sh gen_fewshot.sh $dataset $fewshot
cd ../

for seed in 2023
do
    for sample in 1024
    do  
        # few-shot fine-tuning
        echo "seed: $seed"
        CUDA_VISIBLE_DEVICES=0 python finetune.py \
            --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75  \
            --train_data_path ./data/${dataset}/train/train-${sample}.json \
            --val_data_path ./data/${dataset}/valid/valid-${sample}.json \
            --output_dir ./model/${dataset}/${seed}_${sample} \
            --batch_size 16 \
            --micro_batch_size 2 \
            --num_epochs 10 \
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
        
        # generate
        CUDA_VISIBLE_DEVICES=0 python inference_ddp.py \
            --base_model /root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75  \
            --lora_weights ./model/${dataset}/${seed}_${sample} \
            --result_json_data ./results/${dataset}/${seed}_${sample}.json \
            --dataset ${dataset}

        # evaluate
        cd data/
        PWD=$(pwd)
        echo "current work directory: $PWD"

        gpu_id=0
        res_file=${seed}_${sample}
        sh evaluate.sh ${res_file} ${gpu_id}
    done
done