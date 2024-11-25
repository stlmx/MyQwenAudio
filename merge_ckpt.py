import torch
from peft import PeftModel

def merge_lora_weights(model: torch.nn.Module, lora_model_path: str):
    """
    合并 LoRA 微调后的参数到原始模型权重。
    
    Args:
        model (torch.nn.Module): 原始基础模型（如未加载 LoRA 的预训练模型）。
        lora_model_path (str): 训练后的 LoRA 权重路径。
        
    Returns:
        torch.nn.Module: 权重已合并的模型。
    """
    # 加载 LoRA 微调后的模型
    lora_model = PeftModel.from_pretrained(model, lora_model_path)
    
    # 将 LoRA 参数合并到原始模型
    lora_model.merge_and_unload()

    print("LoRA weights have been merged into the base model.")
    return lora_model.base_model


# 示例使用
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    # 加载原始模型
    base_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Qwen-Aduio-Chat")

    # 指定 LoRA 权重路径
    lora_weights_path = "/root/codes/MyQwenAudio/scripts/save/qwen-audio-chat"

    # 合并 LoRA 权重
    merged_model = merge_lora_weights(base_model, lora_weights_path)
    
    # 保存合并后的模型
    merged_model.save_pretrained("/root/autodl-tmp/merged_model")

    print("Merged model has been saved to ./merged-model.")
