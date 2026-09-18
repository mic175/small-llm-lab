from transformers import pipeline

# 1. 下载并加载一个小型语言模型；device=-1 表示使用 CPU。
generator = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1,
)

# 2. 准备给模型的问题。
messages = [
    {
        "role": "user",
        "content": "Explain what a GPU is in two short sentences.",
    }
]

# 3. 让模型生成回答。
result = generator(
    messages,
    max_new_tokens=32,
    do_sample=False,
)

# 4. 输出新生成的助手消息。
answer = result[0]["generated_text"][-1]["content"]
print(answer)