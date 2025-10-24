# AI Foley Studio Backend

This is a FastAPI-based backend for AI-powered footstep detection and audio generation for video content.

## Features

- 🎬 Video processing with YOLO and MediaPipe
- 👣 Footstep detection using hybrid AI models
- 🎵 Automatic audio generation and synchronization
- 🖼️ Frame analysis for environment detection
- 📊 Export results as CSV/JSON
- 🎥 Generate annotated videos with audio

## Tech Stack

- **FastAPI** - Web framework
- **YOLO** - Object detection
- **MediaPipe** - Pose detection
- **LangChain** - LLM integration
- **OpenCV** - Video processing
- **Librosa** - Audio processing

## API Endpoints

- `POST /api/upload-video` - Upload video for processing
- `POST /api/process/{task_id}` - Start video processing
- `GET /api/status/{task_id}` - Get processing status
- `GET /api/download-video/{task_id}` - Download annotated video
- `GET /api/export-csv/{task_id}` - Export results as CSV
- `GET /api/export-json/{task_id}` - Export results as JSON

## Environment Variables

Create a `.env` file with:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Local Development

```bash
pip install -r requirements.txt
uvicorn link:app --host 0.0.0.0 --port 7860
```

## Deployment

This application is designed to be deployed on Hugging Face Spaces.
