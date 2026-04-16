import whisper
import time
import os
import warnings

# 忽略一些底层的警告信息，保持控制台整洁
warnings.filterwarnings("ignore")

def main():
    # 你的音频文件路径（请确保音频文件与本代码在同一目录下）
    audio_path = "test_audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到文件: {audio_path}")
        print("请将任务二使用剪映导出的音频命名为 test_audio.wav 并放在此文件夹中。")
        return

    print("🚀 正在加载 Whisper 模型 (Base)... 这可能需要几秒钟...")
    # 可以选择 "tiny", "base", "small", "medium", "large"
    # base 模型在速度和准确率之间取得了较好的平衡，适合 PC 本地运行
    model = whisper.load_model("base")

    print(f"🎙️ 开始识别音频: {audio_path}")
    start_time = time.time()
    
    # 执行识别 (使用 fp16=False 以避免在某些不支持半精度的 CPU 上报错)
    result = model.transcribe(audio_path, fp16=False)
    
    end_time = time.time()

    print("\n" + "="*40)
    print("📜 识别结果输出:")
    print("="*40)
    print(result["text"])
    print("\n" + "="*40)
    print(f"⏱️ 处理耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
