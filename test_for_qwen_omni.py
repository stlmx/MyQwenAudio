from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from modeling_qwen_dev import QWenLMHeadOmniModel
from tokenization_qwen import QWenTokenizer



torch.manual_seed(1234)

ckpt_path = "/root/autodl-tmp/merged_model_omni"

tokenizer = QWenTokenizer.from_pretrained(pretrained_model_name_or_path=ckpt_path)
tokenizer.pad_token_id = tokenizer.eod_id

model = QWenLMHeadOmniModel.from_pretrained(pretrained_model_name_or_path=ckpt_path, device_map="cuda").eval()



{"messages": [{"role": "user", "audio": "/root/codes/MyQwenAudio/data/power_omni_demo/power_omni_audio/32_human.mp3", "image": "/root/codes/MyQwenAudio/data/power_omni_demo/power_omni_image/SGJX04881.jpg", "content": ""}, {"role": "assistant", "content": "从图中可以看到，推土机位于电力杆塔下方，且和高压输电线路有一定的距离。根据图像显示，高压输电线路在天空中且相对较高。虽然推土机看似没有直接接触到线路，但在实际施工中，为了确保安全需保持至少10米以上的距离。因此，目前推土机没有确切的触电风险，但需要谨慎操作，保证安全距离。"}]}


# {"messages": [{"role": "user", "audio": "/root/codes/MyQwenAudio/data/power_omni_demo/power_omni_audio/38_human.mp3", "image": "/root/codes/MyQwenAudio/data/power_omni_demo/power_omni_image/SGJX05070.jpg", "content": ""}, {"role": "assistant", "content": "图像中可以看到施工机械（如挖掘机）在靠近输电线路和电力杆塔进行作业。机械设备在操作过程中可能产生较大震动，并且高大的机械臂可能不小心接触到输电线路。如果机械设备过于接近或操作不当，可能会对输电线路造成物理损害，或者在高压电线下作业存在触电危险。此外，输电线路下方的施工开挖也可能导致电力杆塔的根基不稳，进而影响电力设施的稳定性。因此，存在一定的风险，需要施工方特别注意机械操作距离和施工方法，以确保电力设备的安全。"}]}



# 第一轮对话
query = tokenizer.from_list_format([
    {'audio': '/root/codes/MyQwenAudio/data/power_omni_demo/power_omni_audio/32_human.mp3'}, # Either a local path or an url
    {'text': ''},
    {'image': '/root/codes/MyQwenAudio/data/power_omni_demo/power_omni_image/SGJX04881.jpg'}
])

# import ipdb; ipdb.set_trace()

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

# # 第二轮对话
# response, history = model.chat(tokenizer, 'Find the start time and end time of the word "middle classes"', history=history)
# print(response)
# # The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
