from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import subprocess
import os
import soundfile as sf
from datetime import datetime
import tempfile
import pandas as pd
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

# Mock streamlit before importing real.py
import sys


class MockStreamlit:
    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            pass

        return mock_func


sys.modules['streamlit'] = MockStreamlit()

# Import working classes and functions from real.py
from reel import (
    HybridFootstepDetectionPipeline,
    PersonTracker,
    AudioGenerator,
    create_annotated_video,
    merge_audio_with_video
)

# Import your custom modules
from agent import process_video_for_footstep_audio
from sound_agent import main_sound
from qsec import extract_second_audio_librosa

app = FastAPI(title="Footstep Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)


# ==================== Pydantic Models ====================

class ProcessingConfig(BaseModel):
    sensitivity: str = "medium"
    yolo_conf: float = 0.5
    use_hybrid: bool = True
    create_annotated: bool = True
    add_audio: bool = True
    surface_type: str = "concrete"


class FootstepEvent(BaseModel):
    frame: int
    timecode: str
    foot: str
    event: str
    time_seconds: float
    confidence: float


class ProcessingResult(BaseModel):
    task_id: str
    status: str
    progress: float
    events: Optional[List[FootstepEvent]] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    detection_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LiveDetectionConfig(BaseModel):
    sensitivity: str = "medium"
    yolo_conf: float = 0.5


# ==================== Storage ====================

# In-memory storage for tasks
tasks_storage = {}
video_storage = {}


def get_ffmpeg_path():
    """Get FFmpeg path"""
    possible_paths = [
        r"C:\Users\abhiv\OneDrive\Desktop\agentic ai\SoundFeet\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe",
        "ffmpeg"
    ]

    for path in possible_paths:
        if path == "ffmpeg":
            try:
                result = subprocess.run([path, '-version'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
        else:
            if os.path.exists(path):
                return path
    return None


FFMPEG_PATH = get_ffmpeg_path()


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {"message": "Footstep Detection API", "version": "1.0.0"}


@app.post("/api/upload-video")
async def upload_video(
        file: UploadFile = File(...),
        config: Optional[str] = None
):
    """Upload video and create task"""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

    # Save video to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    content = await file.read()
    temp_file.write(content)
    temp_file.close()

    # Parse config
    if config:
        try:
            config_dict = json.loads(config)
        except:
            config_dict = {}
    else:
        config_dict = {}

    processing_config = ProcessingConfig(**config_dict)

    # Create task
    tasks_storage[task_id] = {
        'task_id': task_id,
        'status': 'uploaded',
        'progress': 0.0,
        'video_path': temp_file.name,
        'config': processing_config.dict(),
        'created_at': datetime.now().isoformat()
    }

    return {
        "task_id": task_id,
        "status": "uploaded",
        "message": "Video uploaded successfully"
    }


@app.post("/api/process/{task_id}")
async def process_video(task_id: str, background_tasks: BackgroundTasks):
    """Start processing video"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    if task['status'] == 'processing':
        return {"message": "Task is already being processed"}

    task['status'] = 'processing'
    task['progress'] = 0.0

    background_tasks.add_task(process_video_task, task_id)

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Video processing started"
    }


def process_video_task(task_id: str):
    """Background task for video processing"""
    try:
        task = tasks_storage[task_id]
        config = task['config']
        video_path = task['video_path']

        # Get video info first
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Initialize pipeline using real.py's class
        pipeline = HybridFootstepDetectionPipeline(
            fps=fps,
            sensitivity=config['sensitivity'],
            yolo_conf=config['yolo_conf']
        )

        # Process video using real.py's method
        def progress_callback(progress):
            task['progress'] = progress

        results = pipeline.process_video(video_path, progress_callback)

        # Update task
        task['status'] = 'completed'
        task['progress'] = 1.0
        task['results'] = results
        task['completed_at'] = datetime.now().isoformat()

    except Exception as e:
        task['status'] = 'failed'
        task['error'] = str(e)
        task['failed_at'] = datetime.now().isoformat()


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and progress"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    response = {
        "task_id": task_id,
        "status": task['status'],
        "progress": task['progress']
    }

    if task['status'] == 'completed' and 'results' in task:
        response['results'] = task['results']
    elif task['status'] == 'failed':
        response['error'] = task.get('error')

    return response


@app.post("/api/generate-video/{task_id}")
async def generate_video(task_id: str, background_tasks: BackgroundTasks):
    """Generate annotated video"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Processing not completed")

    if not task.get('results'):
        raise HTTPException(status_code=400, detail="No results available")

    background_tasks.add_task(generate_video_task, task_id)

    return {
        "task_id": task_id,
        "message": "Video generation started"
    }


def generate_video_task(task_id: str):
    """Background task for video generation using real.py's create_annotated_video"""
    try:
        print(f"[DEBUG] Starting video generation for {task_id}")
        task = tasks_storage[task_id]
        results = task['results']
        video_path = task['video_path']
        config = task['config']

        task['video_generating'] = True
        task['video_ready'] = False

        print(f"[DEBUG] Creating annotated video for {task_id}")

        # Generate output path
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='_annotated.mp4')
        annotated_path = temp_file.name
        temp_file.close()

        print(f"[DEBUG] Output video path: {annotated_path}")
        print(f"[DEBUG] Input video path: {video_path}")

        # Use real.py's create_annotated_video function
        def progress_callback(progress):
            task['video_progress'] = progress
            if int(progress * 100) % 10 == 0:
                print(f"[DEBUG] Video generation progress: {progress * 100:.1f}%")

        success = create_annotated_video(
            input_path=video_path,
            events=results['events'],
            output_path=annotated_path,
            use_hybrid=config.get('use_hybrid', True),
            progress_callback=progress_callback
        )

        if not success:
            raise Exception("Video annotation failed")

        # Verify the file was created
        if not os.path.exists(annotated_path):
            raise Exception(f"Annotated video file was not created at {annotated_path}")

        file_size = os.path.getsize(annotated_path)
        print(f"[DEBUG] Annotated video file size: {file_size} bytes")

        if file_size == 0:
            raise Exception("Annotated video file is empty")

        # Update task
        task['annotated_video'] = annotated_path
        task['video_ready'] = True
        task['video_generating'] = False
        task['video_progress'] = 1.0

        print(f"[DEBUG] Video generation completed for {task_id}")
        print(f"[DEBUG] Video file exists: {os.path.exists(annotated_path)}")

    except Exception as e:
        print(f"[ERROR] Video generation failed for {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        task['video_error'] = str(e)
        task['video_ready'] = False
        task['video_generating'] = False


@app.get("/api/video-status/{task_id}")
async def get_video_status(task_id: str):
    """Check if video is ready for download"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    return {
        "task_id": task_id,
        "video_ready": task.get('video_ready', False),
        "video_generating": task.get('video_generating', False),
        "video_progress": task.get('video_progress', 0.0),
        "video_error": task.get('video_error', None)
    }


@app.get("/api/download-video/{task_id}")
async def download_video(task_id: str):
    """Download annotated video"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    print(f"[DEBUG] Download request for {task_id}")
    print(f"[DEBUG] Video ready: {task.get('video_ready')}")
    print(f"[DEBUG] Annotated video path: {task.get('annotated_video')}")

    if not task.get('video_ready'):
        raise HTTPException(status_code=400, detail="Video not ready")

    video_path = task.get('annotated_video')

    if not video_path:
        raise HTTPException(status_code=404, detail="Video path not set")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found at {video_path}")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"annotated_{task_id}.mp4"
    )


@app.get("/api/export-csv/{task_id}")
async def export_csv(task_id: str):
    """Export results as CSV"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    if task['status'] != 'completed' or 'results' not in task:
        raise HTTPException(status_code=400, detail="No results available")

    events = task['results']['events']
    df = pd.DataFrame(events)

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=footsteps_{task_id}.csv"}
    )


@app.get("/api/export-json/{task_id}")
async def export_json(task_id: str):
    """Export results as JSON"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    if task['status'] != 'completed' or 'results' not in task:
        raise HTTPException(status_code=400, detail="No results available")

    return JSONResponse(content=task['results'])


@app.post("/api/generate-audio-video/{task_id}")
async def generate_audio_video(task_id: str, background_tasks: BackgroundTasks):
    """Generate annotated video with footstep audio"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Processing not completed")

    if not task.get('results'):
        raise HTTPException(status_code=400, detail="No results available")

    background_tasks.add_task(generate_audio_video_task, task_id)

    return {
        "task_id": task_id,
        "message": "Audio video generation started"
    }


def generate_audio_video_task(task_id: str):
    """Background task for generating video with audio using real.py's functions"""
    try:
        print(f"[DEBUG] Starting audio video generation for {task_id}")
        task = tasks_storage[task_id]
        results = task['results']
        video_path = task['video_path']
        config = task['config']

        task['audio_video_generating'] = True
        task['audio_video_ready'] = False

        # Step 1: Generate audio track
        print(f"[DEBUG] Generating audio track...")
        audio_gen = AudioGenerator()

        # Get audio file for surface type
        '''surface_type = config.get('surface_type', 'concrete')
        aud_name = process_video_for_footstep_audio(str(video_path))
        aud_path = main_sound(aud_name)
        aud_path = aud_path['default'].replace(".%(ext)s", ".mp3")'''

        aud_path="audio/Footsteps on Gravel Path Outdoor.mp3"

        duration = results['total_frames'] / results['fps']
        audio_track = audio_gen.create_audio_track(
            results['events'],
            aud_path,
            duration
        )

        task['audio_video_progress'] = 0.3

        # Step 2: Create annotated video
        print(f"[DEBUG] Creating annotated video...")
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='_temp.mp4')
        temp_video_path = temp_video.name
        temp_video.close()

        def video_progress(progress):
            task['audio_video_progress'] = 0.3 + (progress * 0.4)  # 30-70%

        success = create_annotated_video(
            input_path=video_path,
            events=results['events'],
            output_path=temp_video_path,
            use_hybrid=config.get('use_hybrid', True),
            progress_callback=video_progress
        )

        if not success:
            raise Exception("Video annotation failed")

        task['audio_video_progress'] = 0.7

        # Step 3: Merge audio with video
        print(f"[DEBUG] Merging audio with video...")
        final_output = tempfile.NamedTemporaryFile(delete=False, suffix='_final.mp4')
        final_output_path = final_output.name
        final_output.close()

        merge_success = merge_audio_with_video(
            temp_video_path,
            audio_track,
            44100,
            final_output_path
        )

        if not merge_success:
            raise Exception("Audio merge failed")

        # Cleanup temp video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        # Verify final file
        if not os.path.exists(final_output_path):
            raise Exception(f"Final video file was not created at {final_output_path}")

        file_size = os.path.getsize(final_output_path)
        print(f"[DEBUG] Final video file size: {file_size} bytes")

        if file_size == 0:
            raise Exception("Final video file is empty")

        # Update task
        task['audio_video_path'] = final_output_path
        task['audio_video_ready'] = True
        task['audio_video_generating'] = False
        task['audio_video_progress'] = 1.0

        print(f"[DEBUG] Audio video generation completed for {task_id}")

    except Exception as e:
        print(f"[ERROR] Audio video generation failed for {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        task['audio_video_error'] = str(e)
        task['audio_video_ready'] = False
        task['audio_video_generating'] = False


@app.get("/api/audio-video-status/{task_id}")
async def get_audio_video_status(task_id: str):
    """Check if audio video is ready for download"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    return {
        "task_id": task_id,
        "audio_video_ready": task.get('audio_video_ready', False),
        "audio_video_generating": task.get('audio_video_generating', False),
        "audio_video_progress": task.get('audio_video_progress', 0.0),
        "audio_video_error": task.get('audio_video_error', None)
    }


@app.get("/api/download-audio-video/{task_id}")
async def download_audio_video(task_id: str):
    """Download video with audio"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_storage[task_id]

    if not task.get('audio_video_ready'):
        raise HTTPException(status_code=400, detail="Audio video not ready")

    video_path = task.get('audio_video_path')

    if not video_path:
        raise HTTPException(status_code=404, detail="Video path not set")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found at {video_path}")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"footsteps_with_audio_{task_id}.mp4"
    )


@app.post("/api/live/capture-floor")
async def capture_floor_frame(file: UploadFile = File(...)):
    """Capture floor frame for live mode"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    session_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    content = await file.read()
    temp_file.write(content)
    temp_file.close()

    tasks_storage[session_id] = {
        'type': 'live',
        'floor_frame': temp_file.name,
        'created_at': datetime.now().isoformat()
    }

    return {
        "session_id": session_id,
        "message": "Floor frame captured"
    }


@app.post("/api/live/detect-frame/{session_id}")
async def detect_frame(session_id: str, file: UploadFile = File(...)):
    """Detect footsteps in a single frame"""
    if session_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read frame
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # TODO: Implement real-time detection
    # This would use the LiveFootstepDetector class from real.py

    return {
        "session_id": session_id,
        "detected": False,
        "message": "Frame processed"
    }

'''
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)'''
