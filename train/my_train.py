import os
import sys
import torch
import json
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field

import sys
sys.path.append("/root/codes/MyQwenAudio")

from tokenization_qwen import QWenTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser

ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
IGNORE_TOKEN_ID = -100

global RANK

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/root/autodl-tmp/hf_home/Qwen-Audio/")

@dataclass
class TrainingArguments(TrainingArguments):
    use_lora : bool = field(default=False)

@dataclass
class DataArguments:
    data_path : str = field(default="/root/codes/MyQwenAudio/data/emov_db/qa_data.jsonl")


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data

def build_model(args):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, padding_side='right', trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eod_id

    return model, tokenizer


class args:
    model_name_or_path = "/root/autodl-tmp/hf_home/Qwen-Audio"


''''这段函数的作用
由于我的数据集是一个jsonl的文件，每一条的sample的格式如下：
a = {"messages": [{"role": "user", "audio": "/root/autodl-tmp/audio_data/neutral_337-364_0359.wav", "content": "说话者在音频中提到了什么？"}, {"role": "assistant", "content": "Earth and gravel seemed to fill the pan. 锅里似乎装满了泥土和沙砾。"}]}
这个函数用于将其转换为qwen audio的标准输出，{"input_ids", "attention_mask", "labels", "audio_info"}

padding和转换为batch的事情，是在collect_fn里面完成
'''

# 我这里写的是单条数据的process，并不是对于整个数据集的jsonl的统一处理
def process(sources, tokenizer: QWenTokenizer, system_prompt="You are a helpful assistant."):
    special_tokens = tokenizer.AUDIO_ST

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_token = tokenizer.encode("\n")


    # system prompt
    system_tokens = tokenizer.encode(text=f'{"system"}\n{system_prompt}', special_tokens=special_tokens)
    system_label = [im_start] + (len(system_tokens) - 3) * [IGNORE_TOKEN_ID] + [im_end] + nl_token
    raw_system_text = IMSTART + system_prompt + IMEND + '\n'

    assert len(system_tokens) == len(system_label)
    "Unknown bugs when encoding system prompt in function 'process'. "

    input_ids = []; labels = []; audio_infos = []

    input_ids.append(system_tokens)
    labels.append(system_label)
    

    for i, ele in enumerate(sources['messages']):

        if i == 0:
            assert ele['role'] == "user"

            "The first sentence must be asked by human."

            audio_info = tokenizer.process_audio(('<audio>' + ele['audio'] + '</audio>'))

            _text_tokens = tokenizer.encode(ele['content'], special_tokens=set(tokenizer.AUDIO_ST), audio_info=audio_info)
            _label_tokens = [im_start] + (len(_text_tokens) - 3) * [IGNORE_TOKEN_ID] + [im_end] + nl_token
            raw_text = IMSTART + ele['content'] + IMEND + '\n'


            audio_infos.append(audio_info)

        else:
            if ele['role'] == 'user':
                audio_info = tokenizer.process_audio(text=ele['audio']) if 'audio' in ele.keys() else None

                # 这里面最开始没有encode"<im_sart>"这样的tokens，所以要加上，然后target算的时候要注意，只有中间的部分是ignore，开始和结束以及换行符需要算loss.                
                _text_tokens_part = tokenizer.encode(f"'user\n'+ {ele['content']}", special_tokens=set(tokenizer.AUDIO_ST), audio_info=audio_info)
                _text_tokens = [im_start] + _text_tokens_part + [im_end] + nl_token
                
                _label_tokens = [im_start] + (len(_text_tokens) - 3) * [IGNORE_TOKEN_ID] + [im_end] + nl_token
                raw_text = IMSTART + ele['content'] + IMEND + '\n'

            else:
                _text_tokens_part = tokenizer.encode(f"'assitant\n' + {ele['content']}", special_tokens=set(tokenizer.AUDIO_ST), audio_info=audio_info)
                _text_tokens = [im_start] + _text_tokens_part + [im_end] + nl_token

                # 这里面的assistant这个词需要mask，只在具体的answer的句子上算loss即可; 所以这里面前面3个tokens不要
                _label_tokens = [im_start] + [IGNORE_TOKEN_ID] * 2 + _text_tokens[3:-2] + [im_end] + nl_token 
                # _label_tokens = [im_start] + (len(_text_tokens) - 3) * [tokenizer.pad_token_id] + [im_end] + nl_tokens
                raw_text = IMSTART + ele['content'] + IMEND + '\n'

        # TODO: 这里只有单轮对话、单段音频输入
        input_ids.append(_text_tokens)
        labels.append(_label_tokens)


        raw_system_text += raw_text
        raw_text_all = raw_system_text

        # 这里的input_ids还不是tensor，因为我不想直接全部padding到max_len
        attention_mask = [[1 if token != tokenizer.pad_token_id else 0 for token in seq] for seq in input_ids]


        for i in range(len(input_ids)):

                assert len(input_ids[i]) == len(labels[i])

    return (input_ids, labels,  audio_infos, raw_text_all,  attention_mask)



class AudioDataset(Dataset):
    def __init__(self, data_path, tokenizer) -> None:
        super().__init__()
        self.data_path =data_path # jsonl
        self.tokenizer = tokenizer
        self.build_dataset()

    def build_dataset(self):
        data = read_jsonl(self.data_path) # single conversation turn
        self.data = data
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        element = self.data[index]
        input_ids, labels, audio_infos, _, attention_mask = process(element, tokenizer=self.tokenizer)
        return {"input_ids": input_ids, "labels": labels, "audio_info": audio_infos, "attention_mask": attention_mask}
    

from typing import Any, Dict, List

def pad_nested_sequence(sequences, pad_value):
    """
    填充嵌套的序列，使每个子序列和整体序列达到一致的长度。

    Args:
        sequences (List[List[List[int]]]): 嵌套的序列。
        pad_value (int): 填充值。

    Returns:
        List[List[int]]: 填充后的序列。
    """
    max_outer_len = max(len(seq) for seq in sequences)  # 外层最大长度
    max_inner_len = max(len(subseq) for seq in sequences for subseq in seq)  # 内层最大长度

    padded = []
    for seq in sequences:
        # 填充每个子序列到内层最大长度
        padded_seq = [subseq + [pad_value] * (max_inner_len - len(subseq)) for subseq in seq]
        # 填充外层序列到外层最大长度
        padded_seq += [[pad_value] * max_inner_len] * (max_outer_len - len(padded_seq))
        padded.append(padded_seq)
    return padded

def pad_sequence(sequences, max_len, pad_value):
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]



class CustomDataCollatorWithAudio(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 提取字段
        input_ids_batch = [feature["input_ids"] for feature in features]
        labels_batch = [feature["labels"] for feature in features]
        attention_mask_batch = [feature["attention_mask"] for feature in features]

        audio_infos = [feature["audio_info"][0] for feature in features]


        # 填充 input_ids 和 labels 的嵌套序列
        input_ids_padded = pad_nested_sequence(input_ids_batch, self.tokenizer.pad_token_id)
        labels_padded = pad_nested_sequence(labels_batch, -100)  # 忽略位置
        attention_mask_padded = pad_nested_sequence(attention_mask_batch, 0)

        # 转换为张量
        input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long)
        labels_tensor = torch.tensor(labels_padded, dtype=torch.long)

        assert input_ids_tensor.shape == labels_tensor.shape


        attention_mask_tensor = torch.tensor(attention_mask_padded, dtype=torch.long)


        if any(audio_infos):

            audio_span_tokens = []
            for x in audio_infos:
                audio_span_tokens.extend(x['audio_span_tokens'])

            
            audio_batch = {
                "input_audios": torch.concat([info['input_audios'] for info in audio_infos if info]),
                "audio_span_tokens": audio_span_tokens,
                "input_audio_lengths": torch.concat([info['input_audio_lengths'] for info in audio_infos if info])
            }

        results = {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask_tensor,
        }

        results['audio_info'] = audio_batch
        
        # 返回批次
        return results



if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    model, tokenizer = build_model(model_args)
    training_dataset = AudioDataset(data_path=data_args.data_path, tokenizer=tokenizer)

    collator = CustomDataCollatorWithAudio(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, train_dataset=training_dataset, data_collator=collator)

    trainer.train()


    
    
