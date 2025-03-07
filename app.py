from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import uvicorn
from pydantic import BaseModel
import os
import torch
import soundfile as sf
import logging
import traceback
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

app = FastAPI()

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = SparkTTS(model_dir, device)
    return model


def run_tts(
    text,
    model:SparkTTS,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    output_format="wav"
):
    """Perform TTS inference and save the generated audio to a temporary file."""
    # Use temporary directory instead of a permanent one
    temp_dir = tempfile.mkdtemp()
    logging.info(f"Using temporary directory: {temp_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    wav_path = os.path.join(temp_dir, f"{timestamp}.wav")
    
    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )

        sf.write(wav_path, wav, samplerate=16000)
    
    logging.info(f"Temporary audio saved at: {wav_path}")
    
    # Convert to MP3 if requested
    if output_format.lower() == "mp3":
        try:
            import subprocess
            mp3_path = os.path.join(temp_dir, f"{timestamp}.mp3")
            subprocess.call(['ffmpeg', '-y', '-i', wav_path, '-acodec', 'libmp3lame', mp3_path])
            logging.info(f"Temporary MP3 audio saved at: {mp3_path}")
            # Remove the WAV file as we'll return the MP3
            os.remove(wav_path)
            return mp3_path, temp_dir
        except Exception as e:
            logging.error(f"Error converting to MP3: {e}")
            # Return WAV if MP3 conversion fails
            return wav_path, temp_dir
    
    return wav_path, temp_dir

# Initialize model
model = initialize_model()

class clone_data(BaseModel):
    text: str
    prompt_text: str = ""
    prompt_wav_upload: str = ""
    prompt_wav_record: str = ""
    output_format: str = "mp3"  # Default to MP3


class create_data(BaseModel):
    text: str
    gender: str = "female"  # Default to female
    pitch: int = 3  # Default to 'moderate' (3 in UI maps to 'moderate')
    speed: int = 3  # Default to 'moderate' (3 in UI maps to 'moderate')
    output_format: str = "mp3"  # Default to MP3


# Define callback function for voice cloning
@app.post("/voice_clone")
async def voice_clone(data: clone_data, background_tasks: BackgroundTasks):
    """
    Clone voice using text and optional prompt speech.
    Returns the audio file directly and then deletes it.
    """
    temp_dir = None
    try:
        prompt_speech = data.prompt_wav_upload if data.prompt_wav_upload else data.prompt_wav_record
        prompt_text_clean = None if len(data.prompt_text) < 2 else data.prompt_text

        audio_output_path, temp_dir = run_tts(
            data.text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech,
            output_format=data.output_format
        )
        
        # Define cleanup function
        def cleanup_temp_files():
            try:
                if os.path.exists(audio_output_path):
                    os.remove(audio_output_path)
                    logging.info(f"Temporary file {audio_output_path} removed successfully")
                if temp_dir and os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
                    logging.info(f"Temporary directory {temp_dir} removed successfully")
            except Exception as e:
                logging.error(f"Error cleaning up temporary files: {e}")
        
        # Add cleanup task to background tasks
        background_tasks.add_task(cleanup_temp_files)
        
        # Return the audio file directly
        return FileResponse(
            path=audio_output_path, 
            media_type=f"audio/{os.path.splitext(audio_output_path)[1][1:]}",
            filename=os.path.basename(audio_output_path)
        )
    except Exception as e:
        # Clean up files if an error occurs
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        logging.error(f"Error in voice_clone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice_creation")
async def voice_creation(data: create_data, background_tasks: BackgroundTasks):
    """
    Create a synthetic voice with adjustable parameters.
    Returns the audio file directly and then deletes it.
    """
    temp_dir = None
    try:
        # Debug info
        logging.info(f"Received request data: text={data.text}, gender={data.gender}, pitch={data.pitch}, speed={data.speed}")
        
        # Map UI values to actual model values
        # First get the string values from LEVELS_MAP_UI
        if data.pitch in LEVELS_MAP_UI:
            pitch_name = LEVELS_MAP_UI[data.pitch]
            logging.info(f"Mapped pitch {data.pitch} to '{pitch_name}'")
        else:
            pitch_name = "moderate"  # Default
            logging.warning(f"Invalid pitch value {data.pitch}, using default 'moderate'")
            
        if data.speed in LEVELS_MAP_UI:
            speed_name = LEVELS_MAP_UI[data.speed]
            logging.info(f"Mapped speed {data.speed} to '{speed_name}'")
        else:
            speed_name = "moderate"  # Default
            logging.warning(f"Invalid speed value {data.speed}, using default 'moderate'")
        
        logging.info(f"Using pitch_name: {pitch_name}, speed_name: {speed_name}")
        
        audio_output_path, temp_dir = run_tts(
            data.text,
            model,
            gender=data.gender,
            pitch=pitch_name,
            speed=speed_name,
            output_format=data.output_format
        )
        
        # Define cleanup function
        def cleanup_temp_files():
            try:
                if os.path.exists(audio_output_path):
                    os.remove(audio_output_path)
                    logging.info(f"Temporary file {audio_output_path} removed successfully")
                if temp_dir and os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
                    logging.info(f"Temporary directory {temp_dir} removed successfully")
            except Exception as e:
                logging.error(f"Error cleaning up temporary files: {e}")
        
        # Add cleanup task to background tasks
        background_tasks.add_task(cleanup_temp_files)
        
        # Return the audio file directly
        return FileResponse(
            path=audio_output_path, 
            media_type=f"audio/{os.path.splitext(audio_output_path)[1][1:]}",
            filename=os.path.basename(audio_output_path)
        )
        
    except Exception as e:
        # Clean up files if an error occurs
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Log detailed error information
        logging.error(f"Error in voice_clone: {e}")
        logging.error(traceback.format_exc())
        
        # Return a more helpful error response
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating audio: {str(e)}. Please check server logs for details."
        )
    except Exception as e:
        # Clean up files if an error occurs
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Log detailed error information
        logging.error(f"Error in voice_creation: {e}")
        logging.error(traceback.format_exc())
        
        # Return a more helpful error response
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating audio: {str(e)}. Please check server logs for details."
        )


@app.get("/health")
async def health_check():
    try:
        # Basic check if model is loaded
        model_status = "loaded" if model is not None else "not loaded"
        
        # Check LEVELS_MAP in SparkTTS
        from cli.SparkTTS import LEVELS_MAP
        levels_map_keys = list(LEVELS_MAP.keys())
        
        return {
            "status": "healthy", 
            "model": model_status,
            "levels_map_ui": str(LEVELS_MAP_UI),
            "levels_map_keys": str(levels_map_keys),
            "cuda_available": torch.cuda.is_available()
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        logging.error(traceback.format_exc())
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=2333, reload=False)