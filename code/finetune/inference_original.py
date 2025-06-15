import sys

import fire
import gradio as gr
import torch
torch.set_num_threads(1)

# 🔧 强制离线模式，使用本地缓存
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ["HF_HOME"] = "/root/autodl-tmp/nku/DEALRec-main/models_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/nku/DEALRec-main/models_cache"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/nku/DEALRec-main/models_cache"

import transformers
import json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig,  LlamaTokenizer
from transformers import LlamaForCausalLM

# data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

import ipdb  


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "/root/autodl-tmp/nku/DEALRec-main/models_cache/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75",
    lora_weights: str = "./model/games/2023_1024",
    dataset: str = "games",
    result_json_data: str = "./results/games/2023_1024_original.json",
    batch_size: int = 4,
    beam_size: int = 2,
    use_small_test: bool = True,
):
    assert (base_model), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    # 🔧 适配小测试集
    if use_small_test:
        test_data_path = f"data/{dataset}/test/test_small.json"
        print(f"📝 使用小测试集: {test_data_path}")
    else:
        test_data_path = f"data/{dataset}/test/test.json"
        print(f"📝 使用完整测试集: {test_data_path}")
    
    print("test data path:", test_data_path)
    
    # 🔧 单GPU设置，移除DDP相关代码
    device_map = "auto"
    device = torch.device("cuda")

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    # 检查LoRA权重路径是否存在
    if os.path.exists(lora_weights):
        print(f"Loading LoRA weights from: {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map=device_map,
            is_trainable=False
        )
    else:
        print(f"Warning: LoRA weights not found at {lora_weights}, using base model only")

    tokenizer.padding_side = "left"

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    class TestData(Dataset):

        def __init__(self, path):
            super().__init__()

            with open(test_data_path, 'r') as f:
                test_data = json.load(f)

            self.instructions = [_['instruction'] for _ in test_data]
            self.inputs = [_['input'] for _ in test_data]
            self.golds = [_['output'] for _ in test_data]

        def __len__(self):
            assert len(self.instructions) == len(self.inputs)
            return len(self.instructions)
        def __getitem__(self, idx):
            return (self.instructions[idx], self.inputs[idx], self.golds[idx])
        
    class TestCollator(object):

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0

            if isinstance(self.tokenizer, LlamaTokenizer):
                # Allow batched inference
                self.tokenizer.padding_side = "left"

        def __call__(self, batch):
            # print(batch)
            # print("** batch length", len(batch))
            instructions = [_[0] for _ in batch]
            inputs = [_[1] for _ in batch]
            golds = [_[2] for _ in batch]
            prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)

            return (inputs, golds)
        
    def evaluate(
        inputs=None,
        num_beams=4,
        max_new_tokens=64,
        **kwargs,
    ):
        # 🔧 与原始inference_ddp.py完全相同的GenerationConfig
        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=1,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        # ipdb.set_trace()
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        # 🔧 与原始inference_ddp.py完全相同的后处理
        output = [_.split('Response:\n')[-1] for _ in output]
        real_outputs = output
        return real_outputs

    # test json file
    with open(test_data_path, 'r') as f:
        test_file = json.load(f)

    # initialize dataset
    test_data = TestData(test_data_path)
    
    # set collator
    test_data = TestData(test_data_path)
    collator = TestCollator(tokenizer)
    
    # set dataloader - 🔧 移除DDP相关设置
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collator,
                             num_workers=1, shuffle=False, pin_memory=False)

    # batch inference
    all_device_output = []
    all_device_target = []
    for i, batch_ in enumerate(tqdm(test_loader)):
        while True:

            try:
                inputs_ = batch_[0].to(device)
                targets = batch_[1]
                output = evaluate(inputs_, num_beams=beam_size)
                break

            except torch.cuda.OutOfMemoryError as e:
                print("Out of memory!")
                beam_size = beam_size -1
                print("Beam:", beam_size)
            except Exception:
                raise RuntimeError

        # 🔧 单GPU情况：直接添加到结果列表
        all_device_output += output
        all_device_target += targets
        
    # 🔧 单GPU情况：直接保存结果
    res = []
    for i, _ in tqdm(enumerate(all_device_target)):
        instance = {}
        instance['predict'] = all_device_output[i]
        instance['output'] = all_device_target[i]
        res.append(instance)
        
    # 创建结果目录
    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    
    try:
        with open(result_json_data, 'w') as f:
            json.dump(res, f, indent=4)
        print(f"✅ 推理结果已保存到: {result_json_data}")
    except:
        with open("original_inference_output.json", 'w') as f:
            json.dump(res, f, indent=4)
        print("✅ 推理结果已保存到: original_inference_output.json")

def generate_prompt(instruction, input=None):
    # 🔧 与原始inference_ddp.py完全相同的prompt格式
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


if __name__ == "__main__":
    fire.Fire(main) 