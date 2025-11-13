from transformers import AutoTokenizer, AutoModel
import torch

# 本地模型路径（你下载好的目录）
model_path = "/root/autodl-tmp/chatglm2-6b"

print("正在加载模型，请稍等...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

print("模型加载完成！开始聊天吧～")
print("输入 exit 退出，输入 clear 清空对话历史。\n")

history = []

while True:
    query = input("你：").strip()
    if query.lower() in ["exit", "quit"]:
        print("再见！")
        break
    if query.lower() == "clear":
        history = []
        print("已清空对话历史。")
        continue

    # 模型推理
    response, history = model.chat(tokenizer, query, history=history)
    print("GLM：", response)
