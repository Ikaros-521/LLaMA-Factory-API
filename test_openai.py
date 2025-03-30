from openai import OpenAI
import sys

# 配置客户端
client = OpenAI(
    base_url='http://127.0.0.1:8000/v1',
    api_key='ikaros'
)

def test_list_models():
    """测试模型列表接口"""
    print("测试模型列表：")
    for model in client.models.list():
        print(f"- {model}")
    print()

def test_chat_completion():
    """测试普通对话"""
    print("测试普通对话：")
    completion = client.chat.completions.create(
        model="Qwen2-1.5B-Chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
    )
    print(f"Assistant: {completion.choices[0].message.content}")
    print()

def test_chat_stream():
    """测试流式对话"""
    print("测试流式对话：")
    sys.stdout.write("Assistant: ")
    completion = client.chat.completions.create(
        model="Qwen2-1.5B-Chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "用简短的话介绍下自己。"}
        ],
        stream=True
    )
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            sys.stdout.write(chunk.choices[0].delta.content)
            sys.stdout.flush()
    print("\n")

def test_chat_with_history():
    """测试带历史记录的对话"""
    print("测试多轮对话：")
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"}
    ]
    
    # 第一轮
    messages.append({"role": "user", "content": "你好"})
    completion = client.chat.completions.create(
        model="Qwen2-1.5B-Chat",
        messages=messages
    )
    reply = completion.choices[0].message.content
    print(f"User: 你好")
    print(f"Assistant: {reply}")
    messages.append({"role": "assistant", "content": reply})
    
    # 第二轮
    messages.append({"role": "user", "content": "我们刚才说了什么？"})
    completion = client.chat.completions.create(
        model="Qwen2-7B-Chat",
        messages=messages
    )
    reply = completion.choices[0].message.content
    print(f"User: 我们刚才说了什么？")
    print(f"Assistant: {reply}")
    print()

if __name__ == "__main__":
    test_list_models()
    test_chat_completion()
    test_chat_stream()
    test_chat_with_history()