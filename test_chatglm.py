import sys
sys.path.append("/root/autodl-tmp/chatglm-6b")  # 确保 Python 能 import 源码

from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch

model_path = "/root/autodl-tmp/chatglm-6b"

# 加载 tokenizer 和模型
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path).half().cuda()
model.eval()

prompt = "帮我写一条提醒：明天上午九点开会"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
