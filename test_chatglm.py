from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🚀 开始加载模型...")

try:
    model_name = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.half().cuda()
    model.eval()
    print("✅ 模型加载成功！")

    prompt = "帮我写一条提醒：明天上午九点开会"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    print("输出：", tokenizer.decode(output[0], skip_special_tokens=True))

except Exception as e:
    print("❌ 出错了：", e)