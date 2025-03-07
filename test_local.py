import runpod
import os
import sys
import json
import base64
from datetime import datetime

# Import from handler.py (确保handler.py在同一目录)
from handler import handler, initialize_model

def save_audio_output(base64_data, content_type):
    """从base64数据保存音频文件到本地，方便播放测试"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    extension = "mp3" if content_type == "audio/mp3" else "wav"
    output_path = f"test_output_{timestamp}.{extension}"
    
    # 解码base64数据并保存到文件
    audio_data = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(audio_data)
    
    print(f"音频保存到: {output_path}")
    return output_path

def test_tts_service():
    """手动测试TTS服务"""
    # 示例1: 标准语音生成测试
    test_job1 = {
        "id": "test_voice_creation",
        "input": {
            "type": "voice_creation",
            "text": "这是一个RunPod无服务器平台上的SparkTTS测试。",
            "gender": "female",
            "pitch": 3,
            "speed": 3,
            "output_format": "mp3"
        }
    }
    
    # 示例2: 语音克隆测试（如果有语音样本的话）
    test_job2 = {
        "id": "test_voice_clone",
        "input": {
            "type": "voice_clone",
            "text": "这是一个使用语音克隆功能的测试。",
            "prompt_text": "这是我的语音样本。",
            # 如果有语音样本文件，可以在这里加载并编码为base64
            "prompt_speech": "",
            "output_format": "mp3"
        }
    }
    
    # 选择要运行的测试
    print("选择要运行的测试:")
    print("1. 标准语音生成")
    print("2. 语音克隆 (需要语音样本)")
    choice = input("输入选项 (1/2): ").strip()
    
    test_job = test_job1 if choice == "1" else test_job2
    
    # 如果选择语音克隆且有语音样本文件
    if choice == "2":
        prompt_file = input("输入语音样本文件路径 (留空跳过): ").strip()
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "rb") as f:
                audio_data = f.read()
                test_job["input"]["prompt_speech"] = base64.b64encode(audio_data).decode('utf-8')
    
    # 运行测试
    print(f"\n开始测试: {test_job['id']}")
    print(f"输入数据: {json.dumps(test_job['input'], ensure_ascii=False)}")
    
    # 处理请求
    result = handler(test_job)
    
    # 检查结果
    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        print("处理成功!")
        audio_path = save_audio_output(result["audio_data"], result["content_type"])
        print(f"请在本地播放 {audio_path} 检查音频质量")

if __name__ == "__main__":
    # 方法1: 命令行参数测试
    if "--test_input" in sys.argv:
        # 使用runpod库处理测试输入
        # 这将自动调用handler函数
        pass
    # 方法2: 本地API服务器
    elif "--rp_serve_api" in sys.argv:
        # 启动本地服务器
        runpod.serverless.start({"handler": handler})
    # 方法3: 交互式测试
    else:
        test_tts_service()