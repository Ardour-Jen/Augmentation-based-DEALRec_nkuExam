#!/usr/bin/env python3
import sys
import fire
import torch
import json
import os
from tqdm import tqdm

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ["HF_HOME"] = "/root/autodl-tmp/nku/DEALRec-main/models_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/nku/DEALRec-main/models_cache"

from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from torch.utils.data import Dataset, DataLoader

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

class TestData(Dataset):
    def __init__(self, test_data_path):
        super().__init__()
        print(f"📖 加载测试数据: {test_data_path}")
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        self.instructions = [item['instruction'] for item in test_data]
        self.inputs = [item['input'] for item in test_data]
        self.golds = [item['output'] for item in test_data]
        
        print(f"✅ 测试数据加载完成，共 {len(self.instructions)} 个样本")

    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        return (self.instructions[idx], self.inputs[idx], self.golds[idx])

class TestCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        instructions = [item[0] for item in batch]
        inputs = [item[1] for item in batch]
        golds = [item[2] for item in batch]
        
        prompts = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        return (inputs, golds)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def main(
    base_model: str = "/root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75",
    lora_weights: str = "./model/games/2023_1024",
    dataset: str = "games",
    result_json_data: str = "./results/games/2023_1024_small.json",
    batch_size: int = 4,
    beam_size: int = 2,
    max_new_tokens: int = 96,
    use_small_test: bool = True
):
    print("🚀 开始快速推理测试...")
    print(f"基础模型: {base_model}")
    print(f"LoRA权重: {lora_weights}")
    print(f"批次大小: {batch_size}")
    
    # 确定测试数据路径
    if use_small_test:
        test_data_path = f"data/{dataset}/test/test_small.json"
        print(f"📝 使用小测试集: {test_data_path}")
    else:
        test_data_path = f"data/{dataset}/test/test.json"
        print(f"📝 使用完整测试集: {test_data_path}")
    
    # 加载tokenizer
    print("🔤 加载tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # 加载模型
    print("🤖 加载基础模型...")
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 加载LoRA权重
    if os.path.exists(lora_weights):
        print(f"🎯 加载LoRA权重: {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto",
            is_trainable=False
        )
    else:
        print(f"⚠️ LoRA权重未找到: {lora_weights}")
        return

    # 设置模型配置
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    print("✅ 模型加载完成！")

    # 推理函数
    def evaluate(inputs, num_beams=2, max_new_tokens=96):
        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.2,
            repetition_penalty=1.05,
            early_stopping=True,
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        
        sequences = generation_output.sequences
        output = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        processed_output = []
        for text in output:
            if 'Response:\n' in text:
                response = text.split('Response:\n')[-1].strip()
            elif 'Response:' in text:
                response = text.split('Response:')[-1].strip()
            elif '### Response:' in text:
                response = text.split('### Response:')[-1].strip()
            else:
                lines = text.strip().split('\n')
                response_lines = []
                skip_prompt = False
                for line in lines:
                    if 'Below is an instruction' in line or '### Instruction:' in line or '### Input:' in line:
                        skip_prompt = True
                        continue
                    if '### Response:' in line:
                        skip_prompt = False
                        continue
                    if not skip_prompt and line.strip():
                        response_lines.append(line.strip())
                
                response = ' '.join(response_lines[-3:]) if response_lines else text[-100:]
            
            response = response.replace('###', '').replace('Instruction:', '').replace('Input:', '').strip()
            processed_output.append(response)
        
        return processed_output

    # 准备数据
    test_data = TestData(test_data_path)
    collator = TestCollator(tokenizer)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collator, shuffle=False)

    # 批量推理
    print(f"🔮 开始推理，共 {len(test_data)} 个样本...")
    all_predictions = []
    all_targets = []
    
    for i, batch in enumerate(tqdm(test_loader, desc="推理进行中")):
        inputs, targets = batch
        inputs = inputs.to(device)
        
        try:
            predictions = evaluate(inputs, num_beams=beam_size, max_new_tokens=max_new_tokens)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # 显示前几个结果
            if i == 0:
                print("\n📋 前3个推理示例:")
                for j in range(min(3, len(predictions))):
                    print(f"样本 {j+1}:")
                    print(f"  预测: {predictions[j][:100]}...")
                    print(f"  真实: {targets[j][:100]}...")
                    print()
                    
        except Exception as e:
            print(f"❌ 推理出错: {e}")
            break

    # 保存结果
    print("💾 保存推理结果...")
    results = []
    for pred, target in zip(all_predictions, all_targets):
        results.append({
            'predict': pred,
            'output': target
        })
    
    # 创建结果目录
    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    
    with open(result_json_data, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 推理完成！")
    print(f"📊 处理样本数: {len(results)}")
    print(f"📄 结果文件: {result_json_data}")
    
    return results

if __name__ == "__main__":
    fire.Fire(main) 