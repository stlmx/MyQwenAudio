import torch
from transformers import AutoModelForCausalLM
from accelerate import dispatch_model

import subprocess


def detect_platform():
    """
    检测当前集群平台，返回字符串：
    - 'huawei' 表示华为平台
    - 'nvidia' 表示 NVIDIA 平台
    - 'unknown' 表示无法检测到，或者非上述两种平台
    """

    # 检测华为平台
    try:
        # 尝试执行 npu-smi info 查看输出
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        if result.returncode == 0:

            return "huawei"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 检测 NVIDIA 平台
    try:
        # 尝试执行 nvidia-smi 查看输出
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        if result.returncode == 0:
            return "nvidia"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 如果都没检测到
    return "unknown"


def _device_map(num_gpus, num_layers):
    per_gpu_layers = (num_layers + 2) / num_gpus

    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': 0,
        'lm_head': num_gpus-1
    }

    used = 1
    gpu_target = 0
    for i in range(num_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus-1 else 1
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(model_name_or_path, num_gpus: int = 2):
    num_devices = torch.cuda.device_count()

    if num_gpus == 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto',
                                                     trust_remote_code=True).eval()
    elif 1 < num_gpus <= num_devices:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='cpu',
                                                     trust_remote_code=True).eval()
        num_layers = model.config.num_hidden_layers
        device_map = _device_map(num_gpus, num_layers)
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
    else:
        raise KeyError

    return model
