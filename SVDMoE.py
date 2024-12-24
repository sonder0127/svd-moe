#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_olmoe import SVD_LlamaAttention, SVD_OlmoeSparseMoeBlock
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from utils.model_utils import *
from utils.method import *
from evaluater import * 
import dill

import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def build_basenet(model_name, model, expert_importance, ratio, dev, top_k=5):
    model.eval()        # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    # for i in tqdm(range(len(layers))):
    for i in range(len(layers)):
        layer = layers[i]

        #### Replace Attn, MLP ####
        if "OLMoE" in model_name:
            #svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_OlmoeSparseMoeBlock(config=model.config, ratio=ratio)
            svd_mlp.gate.weight.data = layer.mlp.gate.weight.data
            # 获取当前层的专家和其重要性分数
            experts = layer.mlp.experts
            layer_importance = expert_importance.get(f'model.layers.{i}.mlp', None)
            if layer_importance is not None and len(experts) > 0:
                # 合并前top_k个专家的参数
                print(f"=======layer {i}=========")
                merged_params = merge_expert_params(experts, layer_importance, top_k) 
                # 创建新的OlmoeMLP层并加载合并后的参数
                svd_mlp.basenet.gate_proj.weight.data = merged_params['gate_proj.weight']
                svd_mlp.basenet.up_proj.weight.data = merged_params['up_proj.weight']
                svd_mlp.basenet.down_proj.weight.data = merged_params['down_proj.weight']
        elif "Qwen" in model_name:
            # 未完成
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_OlmoeSparseMoeBlock(config=model.config, ratio=ratio)
        elif 'deepseekMoE' in model_name:
            # 未完成
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_OlmoeSparseMoeBlock(config=model.config, ratio=ratio)

        #### Replace Attn, MLP ####
        for n in range(len(layer.mlp.experts)):
             # 计算差异并应用到专家的参数
            diff_gate_proj = calculate_param_diff(layer.mlp.experts[n].gate_proj.weight.data, svd_mlp.basenet.gate_proj.weight.data)
            diff_up_proj = calculate_param_diff(layer.mlp.experts[n].up_proj.weight.data, svd_mlp.basenet.up_proj.weight.data)
            diff_down_proj = calculate_param_diff(layer.mlp.experts[n].down_proj.weight.data, svd_mlp.basenet.down_proj.weight.data)

            # 检查是否相等
            gate_proj_equal = torch.allclose(layer.mlp.experts[n].gate_proj.weight.data, svd_mlp.basenet.gate_proj.weight.data + diff_gate_proj)
            up_proj_equal = torch.allclose(layer.mlp.experts[n].up_proj.weight.data, svd_mlp.basenet.up_proj.weight.data + diff_up_proj)
            down_proj_equal = torch.allclose(layer.mlp.experts[n].down_proj.weight.data, svd_mlp.basenet.down_proj.weight.data + diff_down_proj)

            print(f"Expert {n} parameter validation: "
                          f"Gate Proj {'Pass' if gate_proj_equal else 'Fail'}, "
                          f"Up Proj {'Pass' if up_proj_equal else 'Fail'}, "
                          f"Down Proj {'Pass' if down_proj_equal else 'Fail'}")
            
            layer.mlp.experts[n].gate_proj.weight.data = diff_gate_proj
            layer.mlp.experts[n].up_proj.weight.data = diff_up_proj
            layer.mlp.experts[n].down_proj.weight.data = diff_down_proj

            subset = find_layers(layer.mlp.experts[n])
            for name in subset:
                W = subset[name].weight.data.float().to(dev)    # 从子层中提取权重矩阵W并转换为浮点类型
                dtype = W.dtype

                U, S, VT = torch.linalg.svd(W, full_matrices=False)       # 对权重矩阵进行SVD分解
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))    # 计算截断后的奇异值数量
                truc_s = S[:num_s_after_trunc]  # 截断奇异值
                truc_u = U[:, :num_s_after_trunc]   # 截断U矩阵
                truc_v = VT[:num_s_after_trunc, :]    # 计算截断后的V矩阵
                truc_sigma = torch.diag(truc_s)     # 构建对角矩阵
                
                sqrtSigma = torch.sqrt(truc_sigma)      # 计算对角矩阵的平方根
                svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)     # 计算低秩分解后的U矩阵
                svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)     # 计算低秩分解后的V矩阵

                #### Replace Attn, MLP ####
                if 'OLMoE' in model_name:
                    if "q_proj" in name:
                        svd_attn.q_u_proj.weight.data = svd_u
                        svd_attn.q_v_proj.weight.data = svd_v
                    elif "k_proj" in name:
                        svd_attn.k_u_proj.weight.data = svd_u
                        svd_attn.k_v_proj.weight.data = svd_v
                    elif "v_proj" in name:
                        svd_attn.v_u_proj.weight.data = svd_u
                        svd_attn.v_v_proj.weight.data = svd_v
                    elif "o_proj" in name:
                        svd_attn.o_u_proj.weight.data = svd_u
                        svd_attn.o_v_proj.weight.data = svd_v
                        layer.self_attn =  svd_attn
                    elif "gate_proj" in name:
                        svd_mlp.experts[n].gate_u_proj.weight.data = svd_u
                        svd_mlp.experts[n].gate_v_proj.weight.data = svd_v
                    elif "down_proj" in name:
                        svd_mlp.experts[n].down_u_proj.weight.data = svd_u
                        svd_mlp.experts[n].down_v_proj.weight.data = svd_v
                    elif "up_proj" in name:
                        svd_mlp.experts[n].up_u_proj.weight.data = svd_u
                        svd_mlp.experts[n].up_v_proj.weight.data = svd_v
                        # layer.mlp.experts[n] = svd_mlp.experts[n]
                # 清理不再使用的变量
                W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
                del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma

        layer.mlp = svd_mlp
        del layer   # 删除当前层的引用
        torch.cuda.empty_cache()
    print(model)

class MoEModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(MoEModelWrapper, self).__init__()
        self.model = model
        self.expert_counts = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # 为每个MoE层注册一个钩子，用来捕捉专家选择
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):  # 使用实际的MoE层类型
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)  # 保存钩子句柄
                # module.register_forward_hook(self._expert_layer_forward_hook(name))
                print(f"Hook registered for MoE layer: {name}")  # 调试信息

                
        print("register_hooks")

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            print(f"Forward hook triggered for MoE layer: {layer_name}")  # 调试信息
            final_hidden_states, router_logits = output

            # 将 router_logits 大于等于 0 的位置设为 1，其余设为 0
            selected_experts_mask = (router_logits >= 0).to(dtype=torch.float)

            # 将 mask 转换为整数索引
            selected_experts_indices = torch.nonzero(selected_experts_mask, as_tuple=True)[1]

            if layer_name not in self.expert_counts:
                self.expert_counts[layer_name] = torch.zeros(module.num_experts, dtype=torch.int64)

            # 统计每个专家被选中的次数
            unique, counts = torch.unique(selected_experts_indices, return_counts=True)
            self.expert_counts[layer_name][unique] += counts

            # 添加调试信息以确保逻辑正确执行
            print(f"Layer {layer_name} selected experts indices: {selected_experts_indices}")
            print(f"Updated expert counts for layer {layer_name}: {self.expert_counts[layer_name]}")

        return hook

    def reset_counts(self):
        print("reset_counts")
        self.expert_counts = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_expert_importance(self):
        """获取每一层专家的重要性"""
        print("get_expert_importance")
        expert_importance = {}
        for layer, counts in self.expert_counts.items():
            total_calls = counts.sum().item()
            importance = (counts / total_calls).tolist() if total_calls > 0 else [0] * len(counts)
            expert_importance[layer] = importance
        return expert_importance
    
    def remove_hooks(self):
        """移除所有已注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("Hooks removed")


def importance_experts(model, calib_loader, device='cuda'):
    """
    统计 MoE 模型中每个专家层的专家重要性。

    参数:
        model: 要分析的模型实例。
        calib_loader: 数据加载器，提供校准数据。
        device: 设备类型（'cuda' 或 'cpu'），默认为 'cuda'。
    """
    # 包装模型
    wrapped_model = MoEModelWrapper(model)

    # 确保模型处于评估模式
    wrapped_model.eval()

    # 将模型移动到指定设备
    wrapped_model.to(device)

    # 清空之前的计数
    wrapped_model.reset_counts()

    # 统计专家被选中的次数
    with torch.no_grad():  # 不计算梯度
        for batch in tqdm(calib_loader, desc="Calculating expert importance"):
            # 添加调试信息
            print("Batch structure:", batch)
            if isinstance(batch, dict):
                # 如果 batch 是一个字典，则创建一个包含该字典的列表
                batch = [batch]

            # 准备输入数据
            inputs = prepare_inputs_for_model(batch, tokenizer, device=device)

            # 前向传播
            outputs = wrapped_model(**inputs)

    # 获取并打印统计结果
    expert_importance = wrapped_model.get_expert_importance()
    for layer, importance in expert_importance.items():
        print(f"Layer {layer}:")
        print(importance)

    # 移除钩子
    wrapped_model.remove_hooks()
     # 创建 outfiles 目录（如果不存在）
    output_dir = "fig_outfile"
    os.makedirs(output_dir, exist_ok=True)
        # 绘制每一层的专家重要性并保存图表
    for layer, importance in expert_importance.items():
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance, tick_label=[str(i) for i in range(len(importance))])
        plt.title(f"Expert Importance for Layer {layer}")
        plt.xlabel('Expert Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # 保存图表
        filename = os.path.join(output_dir, f"{layer}_expert_importance.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved plot for layer {layer} to {filename}")

    return expert_importance


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='OLMoE', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cpu", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')

    parser.add_argument('--importance_path', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.model == 'OLMoE':
        model_load_path = './model/OLMoE-1B-7B-0924'

    if args.step == 0:
        model, tokenizer = my_get_MoE_model_from_local(args.DEV, model_id=model_load_path)
        model = model.eval()
        if args.importance_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            expert_importance = importance_experts(model, cali_white_data, args.DEV)
            save_expert_importance(expert_importance,'outfile/expert_importance', f'model={args.model}_dataset={args.dataset}_nsamples={args.whitening_nsamples}.json')
        else:
            with open(f'{args.importance_path}/model={args.model}_dataset={args.dataset}_nsamples={args.whitening_nsamples}.json', 'r') as file:
                expert_importance = json.load(file)

        build_basenet(args.model, model, expert_importance, args.ratio, args.DEV)

        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_SVD_only_' + str(args.ratio) + '.pt')
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)  # 从hugging face加载模型
        else:
            print("======load========")
            model, tokenizer = get_model_from_local(args.model_path)
            #model, tokenizer = my_get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        model_test(model, tokenizer, args.DEV)
        exit(0)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)