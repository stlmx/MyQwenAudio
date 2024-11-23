# torchrun --nnode=1 --nproc_per_node=8 /root/codes/MyQwenAudio/train/my_train.py \
#         --output_dir ./tmp \
#         --deepspeed /root/codes/MyQwenAudio/scripts/ds_zero3.json



export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

torchrun --nnode=1 --nproc_per_node=8 /root/codes/MyQwenAudio/train/my_train.py \
        --output_dir ./tmp \
        --fp16 true \
        --learning_rate 1e-5 \
        --logging_steps 1 \
        --weight_decay 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --deepspeed /root/codes/MyQwenAudio/scripts/ds_zero3.json