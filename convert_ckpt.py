import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_qwen_dev import QWenLMHeadOmniModel

class ViTWeightHandler:
    def __init__(self):
        print("Init finished for ViTWeightHandler.")

    def save_vit_weights(self, vit_weights, save_path: str):
        """保存ViT部分权重到指定路径"""
        torch.save(vit_weights, save_path)
        print(f"ViT 权重已保存至 {save_path}")

    def load_vit_weights(self, target_model, load_path: str, load_layers: int = None):
        """从指定路径加载ViT权重，并根据需要加载指定层"""
        vit_weights = torch.load(load_path)
        
        # 如果只加载特定层
        if load_layers is not None:
            vit_weights = self._get_layers(vit_weights, load_layers)

        # 加载到目标模型
        target_model.load_state_dict(vit_weights, strict=True)
        print(f"已将ViT权重加载到目标模型，加载层数：{load_layers or '全部'}")

    def _get_layers(self, vit_weights: dict, load_layers: int):
        """根据需要加载的层数筛选权重"""
        selected_weights = {}
        for idx, (key, value) in enumerate(vit_weights.items()):
            # 提取ViT层的前`load_layers`个权重
            if 'encoder.layer' in key:
                layer_idx = int(key.split('.')[2])  # 假设权重名为 encoder.layer.X
                if layer_idx < load_layers:
                    selected_weights[key] = value
            else:
                selected_weights[key] = value  # 对于不在层中的权重（如Embedding、Position等）
        return selected_weights

def merge_vit_weights_to_audio(qwen_vl_path: str, qwen_audio_path: str, output_path: str, load_layers: int = None):
    # 加载 Qwen_VL 模型并提取 visual 权重
    qwen_vl_model = AutoModelForCausalLM.from_pretrained(qwen_vl_path, trust_remote_code=True)
    vit_handler = ViTWeightHandler()

    vit_weights_path = './tmp/vit_weights.pth'
    
    # 保存 ViT 权重并从 Qwen_VL 中提取
    vit_handler.save_vit_weights(qwen_vl_model.transformer.visual.state_dict(), vit_weights_path)
    
    import ipdb; ipdb.set_trace()
    # 加载 Qwen_Audio_Chat 模型并加载其权重
    qwen_audio_model = QWenLMHeadOmniModel.from_pretrained(qwen_audio_path)
    # qwen_audio_model = AutoModelForCausalLM.from_pretrained(qwen_audio_path)

    # 加载 Qwen_VL 的 ViT 权重到 Qwen_Audio_Chat
    vit_handler.load_vit_weights(qwen_audio_model.transformer.visual, vit_weights_path, load_layers)

    # 保存合并后的模型权重并确保符合 Hugging Face 的目录结构
    qwen_audio_model.save_pretrained(output_path)
    
    # 还可以保存 tokenizer（如果需要的话）
    # tokenizer = AutoTokenizer.from_pretrained(qwen_audio_path)
    # tokenizer.save_pretrained(output_path)
    
    print(f"合并后的模型已保存至 Hugging Face 标准结构路径：{output_path}")

    return qwen_audio_model

# 使用实例：合并 Qwen_VL 的 ViT 权重到 Qwen_Audio_Chat
qwen_vl_model_path = "/root/autodl-tmp/hf_home/Qwen_VL"
qwen_audio_model_path = "/root/autodl-tmp/Qwen-Aduio-Chat"
output_model_path = " /root/autodl-tmp/hf_home/Qwen_Merged_Model"

# 合并 ViT 权重到 audio 模型中
# merged_model = merge_vit_weights_to_audio(qwen_vl_model_path, qwen_audio_model_path, output_model_path, load_layers=48)
print(f"让我们开始测试吧")
model = QWenLMHeadOmniModel.from_pretrained("/root/autodl-tmp/hf_home/Qwen_Merged_Model")
print(model)
