import os
from google import genai

def main():
    # 1. 安全获取配置：从环境变量中读取 API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ 错误：未找到 GEMINI_API_KEY 环境变量。")
        print("👉 请先参考 README.md 中的说明配置你的 API Key。")
        return

    # 2. 初始化 Gemini 客户端
    client = genai.Client(api_key=api_key)
    
    print("🤖 欢迎使用 Gemini Chatbot！(输入 'quit' 或 'exit' 退出)")
    print("-" * 50)
    
    # 3. 循环对话逻辑
    while True:
        user_input = input("\n🧑‍💻 你: ")
        
        # 退出条件
        if user_input.lower() in ['quit', 'exit']:
            print("🤖 再见！")
            break
            
        # 防止空输入
        if not user_input.strip():
            continue
            
        print("🤖 Gemini 思考中...")
        
        try:
            # 4. 调用模型并获取回复 (已指定为 Gemini 3)
            response = client.models.generate_content(
                model='gemini-3.0-pro',
                contents=user_input,
            )
            print(f"🤖 Gemini: {response.text}")
            
        except Exception as e:
            print(f"❌ 调用模型时发生错误: {e}")
            print("💡 提示: 如果你所在的网络无法直连 Google，请确保在终端中配置了 HTTP_PROXY / HTTPS_PROXY 环境变量。")

if __name__ == "__main__":
    main()
