import runpod
import os
import torch
import soundfile as sf
import logging
import base64
import tempfile
from datetime import datetime
import traceback
from io import BytesIO
import sys
import json

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Import your SparkTTS module - adjust the import path as needed
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# Global variable for model
model = None

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return SparkTTS(model_dir, device)

def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    output_format="wav"
):
    """Perform TTS inference and return the audio data."""
    # Use temporary directory for any needed file operations
    temp_dir = tempfile.mkdtemp()
    logging.info(f"Using temporary directory: {temp_dir}")
    
    try:
        if prompt_text is not None:
            prompt_text = None if len(prompt_text) <= 1 else prompt_text
            
        # 如果提供了base64编码的prompt_speech，需要解码
        if prompt_speech and isinstance(prompt_speech, str) and prompt_speech.startswith(("data:", "http:", "https:")) is False:
            try:
                # 尝试将其作为base64解码
                prompt_speech_data = base64.b64decode(prompt_speech)
                # 保存到临时文件
                prompt_speech_path = os.path.join(temp_dir, "prompt.wav")
                with open(prompt_speech_path, "wb") as f:
                    f.write(prompt_speech_data)
                prompt_speech = prompt_speech_path
                logging.info(f"Decoded base64 prompt speech to {prompt_speech_path}")
            except Exception as e:
                logging.error(f"Error decoding prompt_speech: {e}")
                # 如果解码失败，可能是路径或其他格式，直接使用原值

        # Perform inference to get audio waveform
        logging.info("Starting inference...")
        logging.info(f"Parameters: text={text}, gender={gender}, pitch={pitch}, speed={speed}")
        with torch.no_grad():
            wav = model.inference(
                text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
            )
        
        # Create BytesIO object to hold audio data
        audio_buffer = BytesIO()
        
        # Save audio to the buffer
        if output_format.lower() == "mp3":
            # Save wav temporarily first
            temp_wav_path = os.path.join(temp_dir, "temp.wav")
            sf.write(temp_wav_path, wav, samplerate=16000)
            
            # Convert to MP3 using ffmpeg
            try:
                import subprocess
                temp_mp3_path = os.path.join(temp_dir, "temp.mp3")
                subprocess.call(['ffmpeg', '-y', '-i', temp_wav_path, '-acodec', 'libmp3lame', temp_mp3_path])
                
                # Read the MP3 data
                with open(temp_mp3_path, 'rb') as f:
                    audio_data = f.read()
                    
                content_type = "audio/mp3"
            except Exception as e:
                logging.error(f"Error converting to MP3: {e}")
                # Fall back to WAV if MP3 conversion fails
                sf.write(audio_buffer, wav, samplerate=16000, format='WAV')
                audio_buffer.seek(0)
                audio_data = audio_buffer.read()
                content_type = "audio/wav"
        else:
            # Save as WAV
            sf.write(audio_buffer, wav, samplerate=16000, format='WAV')
            audio_buffer.seek(0)
            audio_data = audio_buffer.read()
            content_type = "audio/wav"
            
        # Encode audio data as base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # 为本地测试保存一个音频文件
        if "--save_output" in sys.argv:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_ext = "mp3" if content_type == "audio/mp3" else "wav"
            output_path = f"tts_output_{timestamp}.{file_ext}"
            with open(output_path, "wb") as f:
                f.write(audio_data)
            logging.info(f"Saved audio output to {output_path}")
        
        return {
            "audio_data": audio_base64,
            "content_type": content_type
        }
    
    except Exception as e:
        logging.error(f"Error in run_tts: {e}")
        logging.error(traceback.format_exc())
        raise e
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def handler(job):
    """
    RunPod handler function to process TTS jobs.
    """
    global model
    
    # Initialize model if not already done
    if model is None:
        model = initialize_model()
    
    job_input = job['input']
    job_type = job_input.get('type', 'voice_creation')  # Default to voice creation
    
    try:
        if job_type == 'voice_clone':
            # Handle voice cloning
            result = run_tts(
                text=job_input.get('text', ''),
                model=model,
                prompt_text=job_input.get('prompt_text', ''),
                prompt_speech=job_input.get('prompt_speech', ''),
                output_format=job_input.get('output_format', 'mp3')
            )
        else:
            # Handle voice creation
            # Map UI values to string values if needed
            pitch_value = job_input.get('pitch', 3)
            speed_value = job_input.get('speed', 3)
            
            pitch_name = LEVELS_MAP_UI.get(pitch_value, "moderate")
            speed_name = LEVELS_MAP_UI.get(speed_value, "moderate")
            
            result = run_tts(
                text=job_input.get('text', ''),
                model=model,
                gender=job_input.get('gender', 'female'),
                pitch=pitch_name,
                speed=speed_name,
                output_format=job_input.get('output_format', 'mp3')
            )
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing job: {e}")
        logging.error(traceback.format_exc())
        return {"error": str(e)}

# For generator-style streaming (optional)
def generator_handler(job):
    """
    Optional generator handler for streaming responses.
    This could be used to stream audio in chunks for longer text.
    """
    # For now, just yield the final result since TTS typically returns a complete audio file
    yield handler(job)

if __name__ == "__main__":
    # Setup for local testing
    import sys
    import json
    runpod.serverless.start({
        "handler": handler,
        # Uncomment for generator-style streaming
        # "handler": generator_handler
        
    })
    # if "--test_input" in sys.argv:
    #     test_input_index = sys.argv.index("--test_input")
    #     if test_input_index + 1 < len(sys.argv):
    #         test_input_json = sys.argv[test_input_index + 1]
    #         try:
    #             job = json.loads(test_input_json)
    #             result = handler(job)
    #             print(json.dumps({"output": result}))
                
    #             # 如果指定了保存选项，显示保存路径提示
    #             if "--save_output" in sys.argv:
    #                 print("\n音频文件已保存，请检查上面的日志获取保存路径。")
    #         except json.JSONDecodeError:
    #             print("Error: Invalid JSON in test_input")
    #     else:
    #         print("Error: --test_input requires a JSON string argument")
    # elif "--rp_serve_api" in sys.argv:
    #     # Start the serverless API service for local testing
    #     runpod.serverless.start({
    #         "handler": handler,
    #         # Uncomment for generator-style streaming
    #         # "handler": generator_handler
    #     })
    # else:
    #     # Start the serverless handler for deployment
    #     runpod.serverless.start({
    #         "handler": handler,
    #         # Uncomment for generator-style streaming
    #         # "handler": generator_handler
    #     })