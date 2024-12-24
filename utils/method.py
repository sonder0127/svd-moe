import os
import json
import torch

def merge_expert_params(experts, importance_scores, k):
    """
    根据重要性分数合并前k个专家的参数。
    
    参数:
        experts (list): 包含所有专家的列表。
        importance_scores (list): 每个专家的重要性分数列表。
        k (int): 选择的专家数量。
        
    返回:
        dict: 合并后的参数字典。
    """
    importance_scores_tensor = torch.tensor(importance_scores, device=experts[0].gate_proj.weight.device)

    # 获取前k个专家及其重要性分数
    sorted_indices = torch.argsort(importance_scores_tensor, descending=True)[:k]
    print("=====sorted_indices======", sorted_indices)
    selected_experts = [experts[i] for i in sorted_indices]
    selected_importances = [importance_scores[i] for i in sorted_indices]
    print("=====selected_importances======", selected_importances)
    # 计算总重要性分数用于归一化
    total_importance = sum(selected_importances)

    # 初始化合并后的参数为0
    merged_params = {
        'gate_proj.weight': torch.zeros_like(experts[0].gate_proj.weight.data),
        'up_proj.weight': torch.zeros_like(experts[0].up_proj.weight.data),
        'down_proj.weight': torch.zeros_like(experts[0].down_proj.weight.data)
    }
    # 加权平均合并参数
    for expert, importance in zip(selected_experts, selected_importances):
        for name, param in merged_params.items():
            param += getattr(expert, name.split('.')[0]).weight.data * (importance / total_importance)

    return merged_params

def calculate_diff(layer1, layer2):
    diff_dict = {}
    # Ensure both layers have the same set of parameters
    param_names = set(layer1.state_dict().keys()) & set(layer2.state_dict().keys())
    for name in param_names:
        param1 = layer1.state_dict()[name]
        param2 = layer2.state_dict()[name]
        # Calculate the difference between the two parameters
        diff = param1 - param2
        # Store the difference in the dictionary
        diff_dict[name] = diff
    
    return diff_dict

def calculate_param_diff(expert_param, basenet_param):

    return expert_param - basenet_param  # 计算从basenet_param到expert_param的差异

def save_expert_importance(expert_importance, output_dir='outfiles', filename='expert_importance.json'):
    """
    将专家重要性保存到 JSON 文件中。

    参数:
        expert_importance: 包含每一层专家重要性的字典。
        output_dir: 输出目录，默认为 'outfiles'。
        filename: 输出文件名，默认为 'expert_importance.json'。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整路径
    output_path = os.path.join(output_dir, filename)

    # 保存为 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(expert_importance, f, indent=4)

    print(f"Expert importance saved to {output_path}")
