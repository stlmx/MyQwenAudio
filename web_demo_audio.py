# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio with audio and image inputs."""

from argparse import ArgumentParser
from pathlib import Path
import base64
import copy
import gradio as gr
import os
import re
import secrets
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_qwen_dev import QWenLMHeadOmniModel
from tokenization_qwen import QWenTokenizer
from transformers.generation import GenerationConfig
from pydub import AudioSegment
from PIL import Image

DEFAULT_CKPT_PATH = "/mnt/afs/limingxuan/work_dirs/Qwen_Omni_20241209"


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
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = QWenLMHeadOmniModel.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer


def _parse_text(text):
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
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        history_filter = []
        audio_idx = 1
        image_idx = 1
        pre = ""
        global last_audio, last_image

        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):  # For audio
                last_audio = q[0]
                q = f'Audio {audio_idx}: <audio>{q[0]}</audio>'
                pre += q + '\n'
                audio_idx += 1
            elif isinstance(q, dict) and "image" in q:  # For image
                last_image = q["image"]
                with open(last_image, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode("utf-8")
                pre += f'<img src="data:image/png;base64,{b64_string}" alt="Uploaded Image" width="200"/>'
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""

        history, message = history_filter[:-1], history_filter[-1][0]
        response, history = model.chat(tokenizer, message, history=history)
        
        # 更新聊天记录
        _chatbot[-1] = (_parse_text(query), response)
        task_history[-1] = (query, _parse_text(response))
        print("Qwen-Audio-Chat: " + _parse_text(response))
        return _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def add_mic(history, task_history, file):
        if file is None:
            return history, task_history
        os.rename(file, file + '.wav')
        task_history = task_history + [((file + '.wav',), None)]
        history = history + [((file + '.wav',), None)]
        return history, task_history

    def add_image(history, task_history, image):
        if image is None:
            return history, task_history
        
        # 保存图像到临时文件
        image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        Image.open(image).save(image_path)
        
        # 将图像转换为 base64 编码
        with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")
        
        # 构造 HTML 图像标签
        image_preview_html = f'<img src="data:image/png;base64,{b64_string}" alt="Uploaded Image" width="200"/>'
        
        # 更新 history 和 task_history
        history = history + [(image_preview_html, None)]
        task_history = task_history + [{"image": image_path, "text": None}]
        return history, task_history


    def reset_user_input():
        """Reset the user input field."""
        return gr.update(value="")
    
    def reset_state(task_history):
        """Reset the chat history and task history."""
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
        <p align="center"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/logo.jpg" style="height: 80px"/><p>""")
        gr.Markdown("<center><font size=8>Qwen-Audio-Chat Bot</center>")
        gr.HTML("""
        <style>
        #chatbot .message .text {
            overflow: visible !important;
            white-space: normal !important;
        }
        </style>
        """)
        gr.Markdown("""<center><font size=3>This WebUI now supports audio and image inputs. \
        (本WebUI支持音频与图片输入。)</center>""")
        
        chatbot = gr.Chatbot(label='Qwen-Audio-Chat', elem_classes="control-height", height=750, elem_id="chatbot")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])
        mic = gr.Audio(sources="microphone", type="filepath")
        addimage_btn = gr.UploadButton("🖼️ Upload Image (上传图片)", file_types=["image"])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["audio"])

        mic.change(add_mic, [chatbot, task_history, mic], [chatbot, task_history])
        addimage_btn.upload(add_image, [chatbot, task_history, addimage_btn], [chatbot, task_history])
        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""\
        <font size=2>Note: This demo is governed by the original license of Qwen-Audio. \
        We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
        including hate speech, violence, pornography, deception, etc. \
        (注：本演示受Qwen-Audio的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
        包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=True,
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