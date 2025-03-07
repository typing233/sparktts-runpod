# RunPod SparkTTS Serverless Deployment Guide

This guide will help you deploy your SparkTTS service on RunPod's serverless platform.

## Deployment Steps

### 1. Prepare Your Code

1. Save the handler code to a file named `handler.py` in your project directory.
2. Make sure your directory structure resembles the following:

```
your-project/
├── handler.py                 # RunPod serverless handler
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── cli/                       # Your SparkTTS CLI directory
│   └── SparkTTS.py            # SparkTTS implementation
└── sparktts/                  # SparkTTS package directory
    └── utils/
        └── token_parser.py    # Contains LEVELS_MAP_UI
```

### 2. Create requirements.txt

Ensure your requirements.txt includes:

```
torch
soundfile
runpod
huggingface_hub
```

Plus any other dependencies your TTS implementation requires.

### 3. Docker Build (Optional - for Local Testing)

To build and test your Docker container locally:

```bash
docker build -t spark-tts-serverless .
docker run -p 8000:8000 spark-tts-serverless
```

### 4. Deploy to RunPod

1. Log in to your RunPod account
2. Navigate to the Serverless section
3. Create a new Serverless Template:
   - Select "Upload Docker image" or "Link to GitHub repository"
   - Configure the template with appropriate GPU resources (recommend at least 16GB VRAM)
   - Set any environment variables if needed
4. Deploy the template to create your serverless endpoint

### 5. Testing the Endpoint

You can test your endpoint using the RunPod API or web interface. Here's a sample API request:

```json
// Voice Creation Example
{
  "input": {
    "type": "voice_creation",
    "text": "Hello, this is a test of SparkTTS running on RunPod serverless.",
    "gender": "female",
    "pitch": 3,
    "speed": 3,
    "output_format": "mp3"
  }
}

// Voice Cloning Example
{
  "input": {
    "type": "voice_clone",
    "text": "This is cloned voice synthesis using SparkTTS.",
    "prompt_text": "This is a sample of my voice.",
    "prompt_speech": "(base64 encoded audio data goes here)",
    "output_format": "mp3"
  }
}
```

The response will include base64-encoded audio data you can decode and play.

## Implementation Notes

### Performance Considerations

- The model is loaded once when the handler starts, which saves time on subsequent requests
- For optimal performance, ensure your RunPod endpoint has sufficient GPU memory
- For very long text inputs, consider implementing chunk-based processing using the generator_handler

### Advanced Configuration

You can modify the handler.py to include additional features:

- Batch processing of multiple TTS requests
- Custom voice generation parameters
- Streaming for long text passages (using generator_handler)

### Troubleshooting

- If you encounter CUDA out-of-memory errors, increase the GPU memory allocation
- For model loading issues, verify the model path and ensure all model files are present
- Check the RunPod logs for detailed error information

## Cost Optimization

- RunPod charges based on GPU time usage
- Keep handler code efficient to minimize processing time
- Consider scaling options based on your traffic patterns

## Security Considerations

- Avoid exposing sensitive information in your code or Docker image
- Consider implementation of authentication for your endpoint
- Be aware of licensing requirements for the SparkTTS model