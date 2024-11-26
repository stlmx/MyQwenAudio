
import torch
from transformers import AutoModelForCausalLM

class ViTWeightHandler:
    def __init__(self):
        print("Init finished for ViTWeightHandler.")
        # 加载模型
        # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        # 提取 ViT 部分的权重

        # import ipdb; ipdb.set_trace()
        # self.vit_weights = self.model.transformer.visual.state_dict()
        # self.num_layers = num_layers  # 如果不传，默认加载所有层

    def save_vit_weights(self, save_path: str):
        """保存ViT部分权重到指定路径"""
        # 保存权重到文件
        torch.save(self.vit_weights, save_path)
        print(f"ViT 权重已保存至 {save_path}")

    def load_vit_weights(self, target_model, load_path: str, load_layers: int = None):
        """从指定路径加载ViT权重，并根据需要加载指定层"""
        # 加载权重
        vit_weights = torch.load(load_path)
        # 如果只加载特定层
        if load_layers is not None:
            vit_weights = self._get_layers(vit_weights, load_layers)
        
        # 加载到目标模型
        target_model.visual.load_state_dict(vit_weights, strict=False)
        print(f"已将ViT权重加载到目标模型，加载层数：{load_layers or '全部'}")
    
    def _get_layers(self, vit_weights: dict, load_layers: int):
        """根据需要加载的层数筛选权重"""
        # 假设权重的层名按顺序排列，可以通过切片获取前几层
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
    


# vit_handler = ViTWeightHandler('/root/autodl-tmp/hf_home/Qwen_VL')
# vit_handler.save_vit_weights('/root/autodl-tmp/hf_home/Qwen_VL_ViT/vit_weights.pth')

# # Step 2: 加载权重到目标模型
# * 这里的model应该是一个LMHeadModel，model.model是其中的transformer部分，还有一部分是lm_head
# vit_handler.load_vit_weights(model.model, 'vit_weights.pth', load_layers=6)  # 只加载前6层
