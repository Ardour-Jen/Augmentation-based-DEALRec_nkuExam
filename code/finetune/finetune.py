import os
import sys
from typing import List
import json
import time
from datetime import datetime

# 🔧 强制离线模式，使用本地缓存
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/root/autodl-tmp/nku/DEALRec-main/models_cache'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/nku/DEALRec-main/models_cache'

import numpy as np 
import fire
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

if gpu_name=="NVIDIA RTX A5000": # enable flash attention, we use this for both few-shot fine-tuning and full data fine-tuning for fair comparison
    from fastchat.train.llama2_flash_attn_monkey_patch import (replace_llama_attn_with_flash_attn,)
    replace_llama_attn_with_flash_attn()

import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, TrainerCallback

# 添加自定义日志回调
class DetailedLoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.start_time = time.time()
        
        # 初始化日志文件
        with open(log_file_path, 'w') as f:
            f.write("timestamp,epoch,step,train_loss,eval_loss,learning_rate,epoch_time,total_time\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 提取关键信息
            epoch = logs.get('epoch', state.epoch)
            step = logs.get('step', state.global_step)
            train_loss = logs.get('train_loss', 'N/A')
            eval_loss = logs.get('eval_loss', 'N/A')
            learning_rate = logs.get('learning_rate', 'N/A')
            
            # 计算时间
            total_time = current_time - self.start_time
            epoch_time = total_time / max(epoch, 1) if epoch > 0 else 0
            
            # 写入CSV格式日志
            log_line = f"{timestamp},{epoch:.2f},{step},{train_loss},{eval_loss},{learning_rate},{epoch_time:.2f},{total_time:.2f}\n"
            
            with open(self.log_file_path, 'a') as f:
                f.write(log_line)
            
            # 同时输出到控制台
            print(f"📊 [训练日志] Epoch: {epoch:.2f}, Step: {step}, Train Loss: {train_loss}, Eval Loss: {eval_loss}, LR: {learning_rate}")

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: str = "",  # 改为单个字符串
    val_data_path: str = "",    # 改为单个字符串
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # logging params
    log_file: str = "",  # 添加日志文件参数
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = False  # 强制禁用wandb
    # 禁用wandb相关环境变量
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # model.set_tau(tau)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    train_data_list = []
    val_data_list = []
    
    def normalize_data_types(example):
        """统一数据类型，确保所有字段都是字符串"""
        return {
            "instruction": str(example["instruction"]) if example["instruction"] is not None else "",
            "input": str(example["input"]) if example["input"] is not None else "",
            "output": (
                str(example["output"]) if isinstance(example["output"], str) 
                else str(example["output"][0]) if isinstance(example["output"], list) and len(example["output"]) > 0 
                else ""
            )
        }

    # 加载训练数据
    print(f"Loading training data from: {train_data_path}")
    train_data = load_dataset("json", data_files=train_data_path)
    train_data = train_data.map(normalize_data_types)
    train_data_list.append(train_data)

    # 加载验证数据  
    print(f"Loading validation data from: {val_data_path}")
    val_data = load_dataset("json", data_files=val_data_path)
    val_data = val_data.map(normalize_data_types)
    val_data_list.append(val_data)

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(val_data_list)):
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])
    

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=5,  # 更频繁的日志记录，从8改为5
            logging_first_step=True,  # 记录第一步
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=50,  # 更频繁的评估，从100改为50
            save_strategy="steps",
            save_steps=200,  # 添加保存步数
            output_dir=output_dir,
            save_total_limit=3,  # 保存更多检查点
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # 明确指定最佳模型指标
            greater_is_better=False,  # loss越小越好
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,  # 完全禁用任何报告
            run_name=None,
            logging_dir=os.path.join(output_dir, "logs"),  # 添加日志目录
            dataloader_pin_memory=False,  # 优化数据加载
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=5),
            DetailedLoggingCallback(log_file if log_file else os.path.join(output_dir, 'training_log.csv'))
        ]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""

if __name__ == "__main__":
    fire.Fire(train)