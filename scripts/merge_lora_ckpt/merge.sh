#! /bin/bash

python merge_ckpt.py --base_model_dir /mnt/afs/limingxuan/hf_home/Qwen_Omni \
                    --lora_model_dir /mnt/ssd_data/limingxuan/codes/MyQwenAudio/save/qwen-omni-chat \
                    --output_model_dir /mnt/afs/limingxuan/work_dirs/Qwen_Omni_20241209
