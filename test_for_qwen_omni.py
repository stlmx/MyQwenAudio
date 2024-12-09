from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from modeling_qwen_dev import QWenLMHeadOmniModel
from tokenization_qwen import QWenTokenizer


torch.manual_seed(1234)

ckpt_path = "/mnt/afs/limingxuan/work_dirs/Qwen_Omni_20241209"

tokenizer = QWenTokenizer.from_pretrained(pretrained_model_name_or_path=ckpt_path)
tokenizer.pad_token_id = tokenizer.eod_id

model = QWenLMHeadOmniModel.from_pretrained(pretrained_model_name_or_path=ckpt_path, device_map="cpu").eval()



{"messages": [{"role": "user", "audio": "/mnt/ssd_data/limingxuan/codes/MyQwenAudio/data/power_omni/power_omni_audio/16_human.mp3", "image": "/mnt/ssd_data/limingxuan/codes/MyQwenAudio/data/power_omni/power_omni_image/SGJX02733.jpg", "content": ""}, {"role": "assistant", "content": "图中可以看到挖掘机的机械臂距离高压输电线路较近，存在潜在碰触的风险。如果机械臂升得过高或者挖掘机移动不当，可能会碰触到电力线路，导致短路或者触电事故。这种情况下需要特别注意以下安全隐患：1. 确保挖掘机操作员了解线路的高度和位置。2. 设置安全警戒线，确保挖掘机不会过于靠近线路。3. 实施实时监控和安全指挥，以避免意外发生。"}]}


# 第一轮对话
query = tokenizer.from_list_format([
    {'audio': "/mnt/ssd_data/limingxuan/codes/MyQwenAudio/data/power_omni/power_omni_audio/16_human.mp3"}, # Either a local path or an url
    {'text': ''},
    {'image': "/mnt/ssd_data/limingxuan/codes/MyQwenAudio/data/power_omni/power_omni_image/SGJX02733.jpg"}
])


response, history = model.chat(tokenizer, query=query, history=None)
print(response)
