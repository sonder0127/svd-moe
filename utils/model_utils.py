#coding:utf8
import os
import sys
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def my_get_model_from_local(device, model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quantization_chioce = 0
    if quantization_chioce == 1:
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 检查并设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS token 作为 pad_token

    # 测试分词器以确保 attention_mask 和 position_ids 被正确生成
    test_input = ["Hello, how are you?", "I'm fine, thank you!"]
    test_encoding = tokenizer(
        test_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,  # 如果不需要 token_type_ids
    )

    # 验证 attention_mask 和 position_ids 是否存在
    if 'attention_mask' not in test_encoding:
        raise ValueError("The tokenizer did not generate an attention mask.")
    
    if 'position_ids' not in test_encoding and "opt" not in model_id:
        # 对于非 "opt" 模型，position_ids 是必需的
        raise ValueError("The tokenizer did not generate position IDs.")

    print("Tokenizer correctly generates attention_mask and position_ids.")
    #exit(0)
    return model, tokenizer

def my_get_MoE_model_from_local(device, model_id):
    from transformers import OlmoeForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_chioce = 0
    if quantization_chioce == 1:
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
        )
        
        model = OlmoeForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        model = OlmoeForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,  # 明确指定只使用本地文件
            device_map=device
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    print("ok=================")
    # # 创建一个包含填充的批量输入
    # batch_inputs = [
    #     "Hello, my dog is cute",
    #     "This sentence is much longer and it contains more words than the first one."
    # ]

    # # 使用 tokenizer 处理输入，确保返回 attention_mask
    # inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    # # 保存原始的 attention_mask 并删除它以进行对比
    # original_attention_mask = inputs["attention_mask"].clone()
    # del inputs["attention_mask"]

    # # 获取两个输出并比较
    # outputs_with_mask = model(**inputs, attention_mask=original_attention_mask)
    # outputs_without_mask = model(**inputs)

    # # 比较两个输出是否相同
    # print("Logits are identical:", torch.allclose(outputs_with_mask.logits, outputs_without_mask.logits))

    return model, tokenizer

    
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def model_test(model, tokenizer, dev):
    # 定义一个简单的输入句子
    input_sentence = "Hello, my dog is cute"

    # 使用 tokenizer 处理输入
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(dev)

    # 获取模型的输出
    with torch.no_grad():  # 确保不计算梯度以节省内存和时间
        outputs = model(**inputs)

    # 打印 logits 或其他您感兴趣的输出部分
    print("Model output logits:", outputs.logits)
    
    # 如果是生成任务，可以解码生成的token ids为文本
    if hasattr(outputs, 'logits'):
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        print("Decoded model output:", decoded_text)
    
def prepare_inputs_for_model(batch, tokenizer, device='cuda'):
    """
    准备模型输入数据。

    参数:
        batch: 来自数据加载器的一个批次数据。
        tokenizer: 用于处理文本输入的分词器（在此函数中可能不需要使用）。
        device: 设备类型（'cuda' 或 'cpu'），默认为 'cuda'。

    返回:
        dict: 包含模型所需输入的字典。
    """
    if not batch or not isinstance(batch, list) or not all(isinstance(item, dict) and 'input_ids' in item and 'attention_mask' in item for item in batch):
        raise ValueError("Batch should be a non-empty list of dictionaries containing 'input_ids' and 'attention_mask'.")
    
    # 合并批次中的所有 input_ids 和 attention_mask
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0).to(device)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0).to(device)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }