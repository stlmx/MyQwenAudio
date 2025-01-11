from utils import detect_platform
platform = detect_platform()

if platform == "huawei":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu


elif platform == "nvidia":
    import torch

else:
    raise EnvironmentError("Unsupported platform: cannot import torch.")

import base64
import copy
import gradio as gr
import os
import re
import secrets
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from pydub import AudioSegment
from PIL import Image

# 你自己的模型与tokenizer引用
from transformers import GenerationConfig
from modeling_qwen_dev import QWenLMHeadOmniModel
from tokenization_qwen import QWenTokenizer

DEFAULT_CKPT_PATH = "/home/ma-user/work/lmx/codes/MyQwenAudio/save/qwen-omni-chat_transmission_line_hongwai_lr2e-4_20250110"

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=6006,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = QWenTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side="right",
        use_fast=False,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        # 也可使用 cuda:n 来指定GPU，或 "auto"
        device_map = "cuda"

    model = QWenLMHeadOmniModel.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=torch.float16
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer


def _parse_text(text):
    """
    仅用于前端展示时，对文本进行简单的代码块解析和转义处理，
    防止markdown/html代码片段污染UI。
    """
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args, model, tokenizer):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        """
        核心聊天函数：
        1. 从 task_history 中提取最新的用户输入 query
        2. 将所有（历史 + 当前）用户输入拼接为训练时对应的多模态文本格式 
           (Picture x: <img>...</img>, Audio x: <audio>...</audio>, 文字等)
        3. 调用 model.chat(...) 得到回复
        4. 更新前端显示
        """
        # 取到最新一条 (q, a)
        # 其中 q 可能是字符串、tuple(音频) 或 dict(图片)
        query = task_history[-1][0]  
        
        # ====== 将历史对话+最新query 拼成 "history, message" ======
        # 1) 深拷贝一下，防止边遍历边改
        history_cp = copy.deepcopy(task_history)

        pre = ""             # 用于拼接多模态prompt
        history_filter = []  # 收集 (prompt, reply)
        audio_idx = 1
        image_idx = 1
        
        for i, (q, a) in enumerate(history_cp):
            # 按类型拼装
            if isinstance(q, str):
                # 文本输入，直接拼到 pre
                pre += q

            elif isinstance(q, (tuple, list)):
                # 音频输入： ("/path/to/audio.wav", )
                pre += f"Audio {audio_idx}: <audio>{q[0]}</audio>\n"
                audio_idx += 1

            elif isinstance(q, dict) and "image" in q:
                # 图像输入： {"image": "/path/to/image.png"}
                pre += f"Picture {image_idx}: <img>{q['image']}</img>\n"
                image_idx += 1

            else:
                # 兜底
                pre += str(q)

            # 把 (pre, a) 存进一个filter列表；这样下一轮拼接的时候，pre会清空
            history_filter.append((pre, a))
            pre = ""

        # 2) 此时 history_filter[-1] 就是最新这条 (prompt, reply=None)
        #    model.chat() 需要把历史(除最后一条) 作为“history”，最后那条 prompt 作为“message”
        #    具体写法因你封装的 model.chat(...) 而异，可以根据需要拆分
        #    这里演示：history = history_filter[:-1], message=history_filter[-1][0]
        history, message = history_filter[:-1], history_filter[-1][0]

        # ====== 调用多模态 chat 接口 ======
        response, updated_history = model.chat(
            tokenizer, 
            message, 
            history=history
        )

        # ====== 更新前端的显示 ======
        # 1) 生成“用户侧”显示文字
        if isinstance(query, str):
            user_show = _parse_text(query)
        elif isinstance(query, (tuple, list)):
            user_show = f"[Audio File]: {query[0]}"
        elif isinstance(query, dict) and "image" in query:
            user_show = f"[Picture File]: {query['image']}"
        else:
            user_show = str(query)

        # 2) 模型回复大概率是文本，直接 _parse_text
        _chatbot[-1] = (user_show, _parse_text(response))

        # ====== 更新 task_history ======
        task_history[-1] = (query, response)

        return _chatbot

    def regenerate(_chatbot, task_history):
        """
        如果用户点了 "Regenerate"，则把最后一次问答对重新生成
        """
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        # 把最后一个回答置 None
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        """
        处理文本输入
        """
        if not text.strip():
            return history, task_history, ""
        history = history + [(text, None)]
        task_history = task_history + [(text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        """
        处理文件上传(如音频)
        """
        if file is None:
            return history, task_history
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def add_mic(history, task_history, file):
        """
        处理麦克风录音
        """
        if file is None:
            return history, task_history
        os.rename(file, file + '.wav')
        history = history + [((file + '.wav',), None)]
        task_history = task_history + [((file + '.wav',), None)]
        return history, task_history

    def add_image(history, task_history, image):
        """
        处理图像上传
        """
        if image is None:
            return history, task_history
        
        # 保存到临时文件
        image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        Image.open(image).save(image_path)

        # 仅用于前端展示
        with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")
        image_preview_html = f'<img src="data:image/png;base64,{b64_string}" alt="Picture" width="200"/>'

        # 前端：展示图像
        history = history + [(image_preview_html, None)]
        # 后端：存成 (dict, None)，dict里带 "image"
        task_history = task_history + [({"image": image_path}, None)]
        return history, task_history

    def reset_user_input():
        """清空输入框"""
        return gr.update(value="")

    def reset_state(task_history):
        """清空全部历史对话"""
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align:center'>Qwen-Audio-Chat Demo</h1>")
        
        chatbot = gr.Chatbot(label='Qwen-Audio-Chat', height=750, elem_id="chatbot")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        mic = gr.Audio(sources="microphone", type="filepath")
        addimage_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History")
            submit_btn = gr.Button("🚀 Submit")
            regen_btn = gr.Button("🤔️ Regenerate")
            addfile_btn = gr.UploadButton("📁 Upload File", file_types=["audio"])

        # 绑定事件
        mic.change(add_mic, [chatbot, task_history, mic], [chatbot, task_history])
        addimage_btn.upload(add_image, [chatbot, task_history, addimage_btn], [chatbot, task_history])

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history, query])\
                  .then(predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])

        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""
        <small>
        Note: This is a multi-modal chat demo for Qwen. Please follow the usage license 
        and avoid generating harmful content.
        </small>
        """)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()