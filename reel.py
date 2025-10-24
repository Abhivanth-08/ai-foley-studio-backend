'''aud_name = process_video_for_footstep_audio(temp_video)
aud_dict = main_sound(aud_name)
aud_path = aud_dict['default'].replace(".%(ext)s", ".mp3")'''

import pandas as pd
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import json
import subprocess
import os
import soundfile as sf
from datetime import datetime
import tempfile
from ultralytics import YOLO
from agent import process_video_for_footstep_audio
from sound_agent import main_sound
from qsec import extract_second_audio_librosa
import threading
import queue
import time
from PIL import Image
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)


def get_ffmpeg_path():
    """Get FFmpeg path with multiple fallback options"""
    possible_paths = [
        r"C:\Users\abhiv\OneDrive\Desktop\agentic ai\SoundFeet\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
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

# Streamlit Configuration
st.set_page_config(
    page_title="Hybrid YOLO-MediaPipe Footstep Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        padding: 1rem;
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        color: #155724;
    }
    .hybrid-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin: 1rem 0;
    }
    .live-indicator {
        background: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .ready-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


class LiveFootstepDetector:
    """Real-time footstep detection for live camera feed"""

    def __init__(self, audio_path, sensitivity='medium', yolo_conf=0.5):
        self.audio_path = audio_path
        self.sensitivity = sensitivity
        self.yolo_conf = yolo_conf
        self.running = False
        self.audio_ready = False

        # Load footstep audio
        try:
            self.footstep_audio, self.sample_rate = extract_second_audio_librosa(
                file_path=audio_path,
                target_second=5,
                sample_rate=44100
            )
            self.audio_ready = True
        except Exception as e:
            st.error(f"Failed to load audio: {str(e)}")
            self.audio_ready = False

        # Initialize detection models
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            st.error(f"Failed to initialize models: {str(e)}")
            return

        # Landmark indices
        self.LEFT_HEEL = 29
        self.RIGHT_HEEL = 30

        # Detection thresholds
        self.thresholds = {
            'low': {'prominence': 0.02, 'velocity_threshold': 0.015},
            'medium': {'prominence': 0.015, 'velocity_threshold': 0.012},
            'high': {'prominence': 0.01, 'velocity_threshold': 0.010}
        }[sensitivity]

        # Tracking state
        self.prev_left_y = None
        self.prev_right_y = None
        self.prev_time = None
        self.left_buffer = []
        self.right_buffer = []
        self.buffer_size = 10

        # Audio playback
        self.audio_queue = queue.Queue()
        self.audio_thread = None

    def start_audio_playback(self):
        """Start audio playback thread"""
        if not self.audio_ready:
            return

        def play_audio():
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True
            )

            while self.running:
                try:
                    foot = self.audio_queue.get(timeout=0.1)
                    # Play footstep sound
                    stream.write(self.footstep_audio.astype(np.float32).tobytes())
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio playback error: {e}")

            stream.stop_stream()
            stream.close()
            p.terminate()

        self.audio_thread = threading.Thread(target=play_audio, daemon=True)
        self.audio_thread.start()

    def detect_heel_strike(self, current_y, prev_y, foot_buffer):
        """Detect heel strike based on vertical velocity and position"""
        if prev_y is None:
            return False

        # Calculate vertical velocity (downward is positive)
        velocity = current_y - prev_y

        # Add to buffer
        foot_buffer.append(current_y)
        if len(foot_buffer) > self.buffer_size:
            foot_buffer.pop(0)

        if len(foot_buffer) < 5:
            return False

        # Detect strike: downward movement followed by stabilization
        # Current position is low (heel on ground)
        # Recent movement was downward
        # Velocity is slowing (strike impact)
        recent_velocities = [foot_buffer[i + 1] - foot_buffer[i]
                             for i in range(len(foot_buffer) - 1)]

        avg_velocity = np.mean(recent_velocities[-3:]) if len(recent_velocities) >= 3 else 0

        is_strike = (
                current_y > 0.7 and  # Heel is low in frame
                velocity > self.thresholds['velocity_threshold'] and  # Moving down
                avg_velocity < velocity * 0.5  # Velocity decreasing (impact)
        )

        return is_strike

    def process_frame(self, frame):
        """Process single frame and detect footsteps"""
        if not self.audio_ready:
            return frame, None

        detected_foot = None

        try:
            # YOLO detection
            results = self.yolo_model(frame, conf=self.yolo_conf, classes=[0], verbose=False)

            person_detected = False
            bbox = None

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    person_detected = True
                    box = boxes[0]  # Take first person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Draw YOLO bbox
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (255, 255, 0), 2)
                    break

            # MediaPipe pose estimation
            if person_detected and bbox:
                # Crop to person region with padding
                x1, y1, x2, y2 = bbox
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)

                cropped = frame[y1:y2, x1:x2]
                rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb_frame)

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark

                    # Get heel positions (adjusted to full frame)
                    left_heel = landmarks[self.LEFT_HEEL]
                    right_heel = landmarks[self.RIGHT_HEEL]

                    left_y = (left_heel.y * (y2 - y1) + y1) / frame.shape[0]
                    right_y = (right_heel.y * (y2 - y1) + y1) / frame.shape[0]

                    # Detect strikes
                    left_strike = self.detect_heel_strike(
                        left_y, self.prev_left_y, self.left_buffer
                    )
                    right_strike = self.detect_heel_strike(
                        right_y, self.prev_right_y, self.right_buffer
                    )

                    if left_strike:
                        detected_foot = 'LEFT'
                        self.audio_queue.put('LEFT')
                    elif right_strike:
                        detected_foot = 'RIGHT'
                        self.audio_queue.put('RIGHT')

                    # Update previous positions
                    self.prev_left_y = left_y
                    self.prev_right_y = right_y

                    # Draw skeleton on full frame
                    for landmark in landmarks:
                        x = int((landmark.x * (x2 - x1) + x1))
                        y = int((landmark.y * (y2 - y1) + y1))
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    # Highlight heels
                    left_heel_x = int((left_heel.x * (x2 - x1) + x1))
                    left_heel_y = int((left_heel.y * (y2 - y1) + y1))
                    right_heel_x = int((right_heel.x * (x2 - x1) + x1))
                    right_heel_y = int((right_heel.y * (y2 - y1) + y1))

                    cv2.circle(frame, (left_heel_x, left_heel_y), 8, (0, 255, 0), -1)
                    cv2.circle(frame, (right_heel_x, right_heel_y), 8, (0, 100, 255), -1)

                    if detected_foot:
                        # Show strike indicator
                        heel_x = left_heel_x if detected_foot == 'LEFT' else right_heel_x
                        heel_y = left_heel_y if detected_foot == 'LEFT' else right_heel_y
                        color = (0, 255, 0) if detected_foot == 'LEFT' else (0, 100, 255)

                        cv2.circle(frame, (heel_x, heel_y), 30, color, 3)
                        cv2.putText(frame, f"{detected_foot} STRIKE!",
                                    (heel_x - 50, heel_y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw status
            status_text = "READY" if self.audio_ready else "NO AUDIO"
            status_color = (0, 255, 0) if self.audio_ready else (0, 0, 255)
            cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        except Exception as e:
            print(f"Frame processing error: {e}")

        return frame, detected_foot

    def start(self):
        """Start the detector"""
        self.running = True
        self.start_audio_playback()

    def stop(self):
        """Stop the detector"""
        self.running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)


class HybridFootstepDetectionPipeline:
    """
    Hybrid Detection Pipeline for video files:
    1. YOLO detects person bounding boxes
    2. MediaPipe estimates pose on detected regions
    3. Track footsteps with improved accuracy
    """

    def __init__(self, fps=30, sensitivity='medium', yolo_conf=0.5):
        self.fps = fps
        self.sensitivity = sensitivity
        self.yolo_conf = yolo_conf

        # Initialize YOLO detector
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            st.success("✅ YOLO detector loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ YOLO loading issue: {str(e)}. Downloading model...")
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                st.success("✅ YOLO detector loaded successfully")
            except Exception as e2:
                st.error(f"❌ Failed to load YOLO: {str(e2)}")
                self.yolo_model = None

        # Initialize MediaPipe pose estimator
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            st.success("✅ MediaPipe pose estimator loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize MediaPipe: {str(e)}")
            self.pose = None

        # Landmark indices
        self.LEFT_HEEL = 29
        self.RIGHT_HEEL = 30
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

        # Detection thresholds
        self.thresholds = {
            'low': {'prominence': 0.02, 'min_interval': 0.4},
            'medium': {'prominence': 0.015, 'min_interval': 0.3},
            'high': {'prominence': 0.01, 'min_interval': 0.25}
        }[sensitivity]

        # Tracking state
        self.person_tracker = PersonTracker()

    def detect_person_yolo(self, frame):
        """Detect person using YOLO"""
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(frame, conf=self.yolo_conf, classes=[0], verbose=False)

            person_boxes = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    person_boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

            return person_boxes
        except Exception as e:
            st.warning(f"YOLO detection failed: {str(e)}")
            return []

    def estimate_pose_mediapipe(self, frame, bbox=None):
        """Estimate pose using MediaPipe on specified region"""
        if self.pose is None:
            return None

        try:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)

                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    return None

                rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
                        landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]

                return results
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self.pose.process(rgb_frame)

        except Exception as e:
            return None

    def process_video(self, video_path, progress_callback=None):
        """Process video with hybrid YOLO-MediaPipe pipeline"""

        if self.yolo_model is None or self.pose is None:
            st.error("❌ Detection models not available")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or total_frames <= 0:
            st.error("❌ Invalid video properties")
            cap.release()
            return None

        left_positions = []
        right_positions = []
        detection_confidence = []
        frame_idx = 0

        yolo_detections = 0
        pose_detections = 0

        st.info(f"🔄 Processing with Hybrid Pipeline: {total_frames} frames")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                person_boxes = self.detect_person_yolo(frame)

                if person_boxes:
                    yolo_detections += 1
                    best_box = self.person_tracker.select_best_person(person_boxes, frame_idx)
                    bbox = best_box[:4]
                    results = self.estimate_pose_mediapipe(frame, bbox)

                    if results and results.pose_landmarks:
                        pose_detections += 1
                        landmarks = results.pose_landmarks.landmark

                        left_y = landmarks[self.LEFT_HEEL].y
                        right_y = landmarks[self.RIGHT_HEEL].y
                        conf = (landmarks[self.LEFT_HEEL].visibility +
                                landmarks[self.RIGHT_HEEL].visibility) / 2

                        left_positions.append(left_y)
                        right_positions.append(right_y)
                        detection_confidence.append(conf)
                    else:
                        left_positions.append(np.nan)
                        right_positions.append(np.nan)
                        detection_confidence.append(0.0)
                else:
                    results = self.estimate_pose_mediapipe(frame, bbox=None)

                    if results and results.pose_landmarks:
                        pose_detections += 1
                        landmarks = results.pose_landmarks.landmark

                        left_positions.append(landmarks[self.LEFT_HEEL].y)
                        right_positions.append(landmarks[self.RIGHT_HEEL].y)
                        detection_confidence.append(0.5)
                    else:
                        left_positions.append(np.nan)
                        right_positions.append(np.nan)
                        detection_confidence.append(0.0)

                frame_idx += 1

                if progress_callback and frame_idx % 10 == 0:
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_callback(progress)

        except Exception as e:
            st.error(f"❌ Video processing error: {str(e)}")
            cap.release()
            return None

        cap.release()

        st.info(
            f"📊 YOLO detections: {yolo_detections}/{total_frames} frames ({yolo_detections / total_frames * 100:.1f}%)")
        st.info(
            f"📊 Pose detections: {pose_detections}/{total_frames} frames ({pose_detections / total_frames * 100:.1f}%)")

        if len(left_positions) == 0:
            st.error("❌ No frames processed successfully")
            return None

        try:
            left_series = pd.Series(left_positions).interpolate(method='linear')
            left_series = left_series.bfill().ffill()
            left_positions = left_series.values

            right_series = pd.Series(right_positions).interpolate(method='linear')
            right_series = right_series.bfill().ffill()
            right_positions = right_series.values

            if len(left_positions) > 5:
                window = min(11, len(left_positions) if len(left_positions) % 2 == 1 else len(left_positions) - 1)
                if window >= 3:
                    left_positions = savgol_filter(left_positions, window, 2)
                    right_positions = savgol_filter(right_positions, window, 2)

            left_strikes = self._detect_strikes(left_positions, fps)
            right_strikes = self._detect_strikes(right_positions, fps)

            events = []

            for frame in left_strikes:
                events.append({
                    'frame': int(frame),
                    'timecode': self._frames_to_smpte(frame, fps),
                    'foot': 'LEFT',
                    'event': 'HEEL_STRIKE',
                    'time_seconds': frame / fps,
                    'confidence': detection_confidence[int(frame)] if int(frame) < len(detection_confidence) else 0.5
                })

            for frame in right_strikes:
                events.append({
                    'frame': int(frame),
                    'timecode': self._frames_to_smpte(frame, fps),
                    'foot': 'RIGHT',
                    'event': 'HEEL_STRIKE',
                    'time_seconds': frame / fps,
                    'confidence': detection_confidence[int(frame)] if int(frame) < len(detection_confidence) else 0.5
                })

            events = sorted(events, key=lambda x: x['frame'])

            return {
                'events': events,
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'left_positions': left_positions.tolist() if hasattr(left_positions, 'tolist') else left_positions,
                'right_positions': right_positions.tolist() if hasattr(right_positions, 'tolist') else right_positions,
                'detection_stats': {
                    'yolo_detections': yolo_detections,
                    'pose_detections': pose_detections,
                    'total_frames': total_frames
                }
            }

        except Exception as e:
            st.error(f"❌ Data processing error: {str(e)}")
            return None

    def _detect_strikes(self, positions, fps):
        """Detect heel strikes from position data"""
        try:
            peaks, _ = find_peaks(
                positions,
                prominence=self.thresholds['prominence'],
                distance=int(fps * self.thresholds['min_interval']),
                height=0.7
            )
            return peaks
        except Exception as e:
            st.warning(f"Peak detection failed: {str(e)}")
            return np.array([])

    def _frames_to_smpte(self, frame, fps):
        """Convert frame number to SMPTE timecode"""
        total_seconds = frame / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int((total_seconds * fps) % fps)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


class PersonTracker:
    """Track person across frames for consistency"""

    def __init__(self, iou_threshold=0.3):
        self.tracked_box = None
        self.last_frame = -1
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def select_best_person(self, person_boxes, frame_idx):
        """Select best person box for tracking consistency"""
        if not person_boxes:
            return None

        if self.tracked_box is not None and frame_idx - self.last_frame < 10:
            max_iou = 0
            best_box = None

            for box in person_boxes:
                iou = self.calculate_iou(self.tracked_box, box)
                if iou > max_iou:
                    max_iou = iou
                    best_box = box

            if max_iou > self.iou_threshold:
                self.tracked_box = best_box
                self.last_frame = frame_idx
                return best_box

        best_box = max(person_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]) * x[4])
        self.tracked_box = best_box
        self.last_frame = frame_idx
        return best_box


class AudioGenerator:
    """Generate footstep audio"""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def generate_footstep(self, aud_path):
        arr, rate = extract_second_audio_librosa(
            file_path=aud_path,
            target_second=5,
            sample_rate=self.sample_rate
        )
        return arr

    def create_audio_track(self, events, aud_path, duration=0.3):
        total_samples = int(duration * self.sample_rate)
        audio_track = np.zeros(total_samples, dtype=np.float32)

        for i, event in enumerate(events):
            step_sound = self.generate_footstep(aud_path)
            pitch_shift = 1.0 + (i % 5 - 2) * 0.03
            indices = np.arange(len(step_sound)) * pitch_shift
            indices = np.clip(indices, 0, len(step_sound) - 1).astype(int)
            step_sound = step_sound[indices]

            start_sample = int(event['time_seconds'] * self.sample_rate)
            end_sample = min(start_sample + len(step_sound), total_samples)
            sound_len = end_sample - start_sample

            if sound_len > 0:
                audio_track[start_sample:end_sample] += step_sound[:sound_len]

        max_val = np.max(np.abs(audio_track))
        if max_val > 0:
            audio_track = audio_track / max_val * 0.8

        return audio_track


def create_annotated_video(input_path, events, output_path, use_hybrid=True, progress_callback=None):
    """Create annotated video with hybrid detection visualization"""

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            st.error("❌ Could not open input video file")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            st.error("❌ Could not create output video file")
            cap.release()
            return False

        event_frames = {e['frame']: e for e in events}

        if use_hybrid:
            yolo_model = YOLO('yolov8n.pt')
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            yolo_model = None
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if use_hybrid and yolo_model:
                    results = yolo_model(frame, conf=0.5, classes=[0], verbose=False)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                          (255, 255, 0), 2)
                            cv2.putText(frame, f'YOLO: {conf:.2f}',
                                        (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=2
                        )
                    )

                if frame_idx in event_frames:
                    event = event_frames[frame_idx]

                    banner_height = 100
                    cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 0, 0), -1)

                    text = f"{event['foot']} HEEL STRIKE"
                    color = (0, 255, 0) if event['foot'] == 'LEFT' else (0, 100, 255)

                    cv2.putText(frame, text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                    if 'confidence' in event:
                        conf_text = f"Conf: {event['confidence']:.2f}"
                        cv2.putText(frame, conf_text, (50, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    circle_x = 50 if event['foot'] == 'LEFT' else width - 50
                    cv2.circle(frame, (circle_x, height - 100), 40, color, -1)

                if use_hybrid:
                    cv2.rectangle(frame, (width - 250, 10), (width - 10, 50), (102, 126, 234), -1)
                    cv2.putText(frame, "HYBRID MODE", (width - 240, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                time_seconds = frame_idx / fps
                hours = int(time_seconds // 3600)
                minutes = int((time_seconds % 3600) // 60)
                seconds = int(time_seconds % 60)
                frame_num = int((time_seconds * fps) % fps)
                timecode = f"TC: {hours:02d}:{minutes:02d}:{seconds:02d}:{frame_num:02d}"

                cv2.rectangle(frame, (0, height - 80), (400, height), (0, 0, 0), -1)
                cv2.putText(frame, timecode, (10, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, height - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                out.write(frame)
                frame_idx += 1

                if progress_callback and frame_idx % 5 == 0:
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_callback(progress)

            except Exception as e:
                st.warning(f"⚠️ Error processing frame {frame_idx}: {str(e)}")
                frame_idx += 1
                continue

        cap.release()
        out.release()
        pose.close()

        return True

    except Exception as e:
        st.error(f"❌ Video annotation failed: {str(e)}")
        try:
            cap.release()
            out.release()
            pose.close()
        except:
            pass
        return False


def merge_audio_with_video(video_path, audio_track, sample_rate, output_path):
    """Merge audio with video using FFmpeg"""

    temp_audio = tempfile.mktemp(suffix='.wav')
    sf.write(temp_audio, audio_track, sample_rate)

    ffmpeg_cmd = FFMPEG_PATH if FFMPEG_PATH else "ffmpeg"

    cmd = [
        ffmpeg_cmd, '-y',
        '-i', str(video_path),
        '-i', temp_audio,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        str(output_path)
    ]

    try:
        if FFMPEG_PATH is None:
            st.warning("FFmpeg not found. Using fallback method.")
            return None

        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        return True

    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        st.error("FFmpeg timed out")
        return False
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def live_streaming_mode():
    """Live streaming mode with frame capture and real-time detection"""

    st.markdown('<h2>📹 Live Streaming Mode</h2>', unsafe_allow_html=True)
    st.info("🎥 This mode allows real-time footstep detection with your device camera")

    # Initialize session state
    if 'floor_frame_captured' not in st.session_state:
        st.session_state.floor_frame_captured = False
    if 'audio_downloaded' not in st.session_state:
        st.session_state.audio_downloaded = False
    if 'live_audio_path' not in st.session_state:
        st.session_state.live_audio_path = None
    if 'live_detector' not in st.session_state:
        st.session_state.live_detector = None
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    # Step 1: Capture floor frame
    st.markdown("### Step 1: Capture Floor Frame 📸")
    st.write("Capture a single frame showing the floor surface for audio analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Camera input for frame capture
        camera_image = st.camera_input("Capture floor image", key="floor_capture")

        if camera_image is not None and not st.session_state.floor_frame_captured:
            # Save captured frame
            image = Image.open(camera_image)
            temp_frame_path = tempfile.mktemp(suffix='.jpg')
            image.save(temp_frame_path)
            st.session_state.floor_frame_path = temp_frame_path

            # Display captured frame
            st.image(image, caption="Captured Floor Frame", use_container_width=True)

            if st.button("✅ Confirm Floor Capture", type="primary", use_container_width=True):
                st.session_state.floor_frame_captured = True
                st.success("✅ Floor frame captured successfully!")
                st.rerun()

    with col2:
        if st.session_state.floor_frame_captured:
            st.markdown('<div class="success-box">✅ Floor Captured</div>', unsafe_allow_html=True)
        else:
            st.info("📸 Capture floor frame to proceed")

    # Step 2: Analyze and download audio
    if st.session_state.floor_frame_captured and not st.session_state.audio_downloaded:
        st.markdown("---")
        st.markdown("### Step 2: Analyze Floor & Download Audio 🔊")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("🔍 Analyze Floor & Generate Audio", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing floor surface and generating audio..."):
                    try:
                        # Create temporary video from frame for processing
                        temp_video = tempfile.mktemp(suffix='.mp4')

                        # Create 1-second video from the captured frame
                        img = cv2.imread(st.session_state.floor_frame_path)
                        height, width = img.shape[:2]

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_video, fourcc, 30, (width, height))

                        # Write 30 frames (1 second at 30fps)
                        for _ in range(30):
                            out.write(img)
                        out.release()

                        # Process video for footstep audio
                        st.info("🎵 Generating footstep audio based on floor analysis...")

                        aud_path="audio/Footsteps on Gravel Path Outdoor.mp3"

                        st.session_state.live_audio_path = aud_path
                        st.session_state.audio_downloaded = True

                        # Clean up temp video
                        if os.path.exists(temp_video):
                            os.remove(temp_video)

                        st.success("✅ Audio generated successfully!")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error generating audio: {str(e)}")

        with col2:
            st.info("🎵 Audio will be generated based on floor type")

    # Step 3: Initialize live detector
    if st.session_state.audio_downloaded and st.session_state.live_detector is None:
        st.markdown("---")
        st.markdown("### Step 3: Initialize Live Detection 🚀")

        col1, col2 = st.columns([2, 1])

        with col1:
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=['low', 'medium', 'high'],
                value='medium'
            )

            yolo_conf = st.slider(
                "YOLO Confidence",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )

            if st.button("🎬 Initialize Live Detector", type="primary", use_container_width=True):
                with st.spinner("⚙️ Initializing detector..."):
                    try:
                        detector = LiveFootstepDetector(
                            audio_path=st.session_state.live_audio_path,
                            sensitivity=sensitivity,
                            yolo_conf=yolo_conf
                        )
                        st.session_state.live_detector = detector
                        st.success("✅ Live detector initialized!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to initialize detector: {str(e)}")

        with col2:
            st.info("🤖 Configure detection parameters")

    # Step 4: Start live detection
    if st.session_state.live_detector is not None:
        st.markdown("---")
        st.markdown('<div class="ready-badge">✅ SYSTEM READY</div>', unsafe_allow_html=True)
        st.markdown("### Step 4: Live Detection 🎯")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("📹 **Camera is ready for live footstep detection**")
            st.write("🚶 Walk in front of the camera and hear footsteps in real-time!")

            # Start/Stop controls
            col_a, col_b = st.columns(2)

            with col_a:
                if not st.session_state.camera_active:
                    if st.button("▶️ Start Live Detection", type="primary", use_container_width=True):
                        st.session_state.camera_active = True
                        st.session_state.live_detector.start()
                        st.rerun()

            with col_b:
                if st.session_state.camera_active:
                    if st.button("⏹️ Stop Detection", type="secondary", use_container_width=True):
                        st.session_state.camera_active = False
                        st.session_state.live_detector.stop()
                        st.rerun()

        with col2:
            if st.session_state.camera_active:
                st.markdown('<div class="live-indicator">🔴 LIVE</div>', unsafe_allow_html=True)
            else:
                st.info("⏸️ Paused")

        # Live video feed
        if st.session_state.camera_active:
            st.markdown("---")

            FRAME_WINDOW = st.image([])

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Cannot access camera. Please check permissions.")
                st.session_state.camera_active = False
            else:
                st.info("📹 Live feed active - Walk to generate footsteps!")

                # Statistics
                step_counter = st.empty()
                left_steps = 0
                right_steps = 0

                try:
                    while st.session_state.camera_active:
                        ret, frame = cap.read()

                        if not ret:
                            st.error("❌ Failed to read from camera")
                            break

                        # Process frame
                        processed_frame, detected_foot = st.session_state.live_detector.process_frame(frame)

                        # Update counters
                        if detected_foot == 'LEFT':
                            left_steps += 1
                        elif detected_foot == 'RIGHT':
                            right_steps += 1

                        # Display frame
                        FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

                        # Update statistics
                        step_counter.metric("Total Steps Detected", left_steps + right_steps,
                                            f"L: {left_steps} | R: {right_steps}")

                        # Check if user stopped
                        if not st.session_state.camera_active:
                            break

                        time.sleep(0.033)  # ~30 FPS

                except Exception as e:
                    st.error(f"❌ Error during live detection: {str(e)}")

                finally:
                    cap.release()
                    st.session_state.live_detector.stop()

        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.floor_frame_captured = False
            st.session_state.audio_downloaded = False
            st.session_state.live_audio_path = None
            st.session_state.live_detector = None
            st.session_state.camera_active = False
            st.rerun()


def video_upload_mode():
    """Original video upload mode"""

    st.markdown('<h2>📤 Video Upload Mode</h2>', unsafe_allow_html=True)

    # Sidebar configuration
    sensitivity = st.sidebar.select_slider(
        "Footstep Sensitivity",
        options=['low', 'medium', 'high'],
        value='medium',
        help="Higher sensitivity detects more subtle footsteps"
    )

    yolo_conf = st.sidebar.slider(
        "YOLO Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Confidence threshold for YOLO person detection"
    )

    surface_type = st.sidebar.selectbox(
        "Surface Type",
        ['concrete', 'wood', 'grass', 'gravel', 'metal'],
        help="Select surface for audio generation"
    )

    use_hybrid = st.sidebar.checkbox(
        "Enable Hybrid Mode",
        value=True,
        help="Use YOLO for person detection + MediaPipe for pose estimation"
    )

    create_annotated = st.sidebar.checkbox("Create Annotated Video", value=True)
    add_audio = st.sidebar.checkbox("Add Footstep Audio", value=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to detect footsteps"
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📹 Input Video")
            st.video(video_path)

        with col2:
            st.subheader("ℹ️ Video Info")
            cap = cv2.VideoCapture(video_path)
            video_info = {
                "Duration": f"{cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS):.2f}s",
                "FPS": f"{cap.get(cv2.CAP_PROP_FPS):.2f}",
                "Resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                "Frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            cap.release()

            for key, value in video_info.items():
                st.metric(key, value)

            if use_hybrid:
                st.success("🤖 Hybrid Mode Active")
            else:
                st.info("📊 MediaPipe Only")

        st.markdown("---")

        if st.button("🚀 Process Video", type="primary", use_container_width=True):

            if use_hybrid:
                st.info("🔄 Running Hybrid YOLO-MediaPipe Pipeline...")
                pipeline = HybridFootstepDetectionPipeline(
                    fps=float(video_info["FPS"]),
                    sensitivity=sensitivity,
                    yolo_conf=yolo_conf
                )
            else:
                st.info("🔄 Running MediaPipe-Only Pipeline...")
                pipeline = HybridFootstepDetectionPipeline(
                    fps=float(video_info["FPS"]),
                    sensitivity=sensitivity,
                    yolo_conf=yolo_conf
                )

            with st.spinner("🔍 Detecting footsteps..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(val):
                    progress_bar.progress(val)
                    status_text.text(f"Processing: {int(val * 100)}%")

                results = pipeline.process_video(video_path, update_progress)
                st.session_state['results'] = results
                st.session_state['video_path'] = video_path
                st.session_state['use_hybrid'] = use_hybrid

                progress_bar.empty()
                status_text.empty()

            if results:
                st.markdown('<div class="success-box">✅ Footstep detection complete!</div>',
                            unsafe_allow_html=True)
                st.success(f"Detected **{len(results['events'])}** footstep events")

                if 'detection_stats' in results:
                    stats = results['detection_stats']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("YOLO Detections",
                                f"{stats['yolo_detections']}/{stats['total_frames']}")
                    col2.metric("Pose Detections",
                                f"{stats['pose_detections']}/{stats['total_frames']}")
                    col3.metric("Success Rate",
                                f"{stats['pose_detections'] / stats['total_frames'] * 100:.1f}%")

        # Display results (existing code continues...)
        if 'results' in st.session_state:
            results = st.session_state['results']

            st.markdown("---")
            st.subheader("📊 Detection Results")

            col1, col2, col3, col4 = st.columns(4)

            left_count = len([e for e in results['events'] if e['foot'] == 'LEFT'])
            right_count = len([e for e in results['events'] if e['foot'] == 'RIGHT'])
            avg_cadence = len(results['events']) / (results['total_frames'] / results['fps']) * 60
            avg_conf = np.mean([e.get('confidence', 0.5) for e in results['events']])

            col1.metric("Total Events", len(results['events']))
            col2.metric("Left Foot", left_count)
            col3.metric("Right Foot", right_count)
            col4.metric("Avg Confidence", f"{avg_conf:.2f}")

            st.metric("Average Cadence", f"{avg_cadence:.1f} steps/min")

            st.subheader("📋 Detected Events")
            events_df = pd.DataFrame(results['events'])

            if not events_df.empty:
                st.dataframe(
                    events_df.style.apply(
                        lambda x: ['background-color: #e8f5e9' if x.foot == 'LEFT'
                                   else 'background-color: #fff3e0' for _ in x],
                        axis=1
                    ),
                    use_container_width=True,
                    height=300
                )

            st.subheader("💾 Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv = events_df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv,
                    f"footsteps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                json_data = json.dumps(results['events'], indent=2)
                st.download_button(
                    "📋 Download JSON",
                    json_data,
                    f"footsteps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )

            with col3:
                timecode_text = "\n".join([
                    f"{e['timecode']}\t{e['foot']}\t{e['event']}\t{e.get('confidence', 0.5):.2f}"
                    for e in results['events']
                ])
                st.download_button(
                    "⏱️ Download Timecode",
                    timecode_text,
                    f"timecode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("🎥 Generate Output Video")

            col1, col2 = st.columns(2)

            with col1:
                if create_annotated and st.button("Create Annotated Video", use_container_width=True):
                    with st.spinner("Creating annotated video..."):
                        annotated_path = tempfile.mktemp(suffix='_annotated.mp4')
                        progress_bar = st.progress(0)

                        success = create_annotated_video(
                            st.session_state['video_path'],
                            results['events'],
                            annotated_path,
                            use_hybrid=st.session_state.get('use_hybrid', False),
                            progress_callback=lambda v: progress_bar.progress(v)
                        )

                        if success:
                            st.session_state['annotated_video'] = annotated_path
                            progress_bar.empty()
                            st.success("✅ Annotated video ready!")
                        else:
                            st.error("❌ Failed to create annotated video")

            with col2:
                if add_audio and st.button("Generate with Audio", use_container_width=True):
                    with st.spinner("Generating audio and merging..."):
                        audio_gen = AudioGenerator()
                        aud_path="audio/Footsteps on Gravel Path Outdoor.mp3"
                        duration = results['total_frames'] / results['fps']
                        audio_track = audio_gen.create_audio_track(
                            results['events'],
                            aud_path,
                            duration
                        )

                        temp_video = tempfile.mktemp(suffix='_temp.mp4')
                        progress_bar = st.progress(0)

                        create_annotated_video(
                            st.session_state['video_path'],
                            results['events'],
                            temp_video,
                            use_hybrid=st.session_state.get('use_hybrid', False),
                            progress_callback=lambda v: progress_bar.progress(v * 0.7)
                        )

                        final_output = tempfile.mktemp(suffix='_final.mp4')
                        success = merge_audio_with_video(
                            temp_video,
                            audio_track,
                            44100,
                            final_output
                        )

                        progress_bar.progress(1.0)
                        progress_bar.empty()

                        if success:
                            st.session_state['final_video'] = final_output
                            st.success("✅ Video with audio ready!")
                        else:
                            st.error("❌ Failed to merge audio")

            if 'annotated_video' in st.session_state:
                st.markdown("---")
                st.subheader("📺 Annotated Video")
                st.video(st.session_state['annotated_video'])

                with open(st.session_state['annotated_video'], 'rb') as f:
                    st.download_button(
                        "📥 Download Annotated Video",
                        f,
                        f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        "video/mp4",
                        use_container_width=True
                    )

            if 'final_video' in st.session_state:
                st.markdown("---")
                st.subheader("🔊 Final Video with Audio")
                st.video(st.session_state['final_video'])

                with open(st.session_state['final_video'], 'rb') as f:
                    st.download_button(
                        "📥 Download Final Video",
                        f,
                        f"final_with_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        "video/mp4",
                        use_container_width=True
                    )


def main():
    st.markdown('<h1 class="main-header">🎬 Hybrid YOLO-MediaPipe Footstep Detection</h1>',
                unsafe_allow_html=True)
    st.markdown('<div class="hybrid-badge">🚀 YOLO Person Detection + MediaPipe Pose Estimation</div>',
                unsafe_allow_html=True)
    st.markdown("### Advanced AI-Powered Foley Tool with Dual-Stage Detection Pipeline")

    # Mode selection
    st.markdown("---")
    st.markdown("## 🎯 Select Mode")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Video Upload Mode", use_container_width=True, type="primary"):
            st.session_state.mode = 'upload'

    with col2:
        if st.button("📹 Live Streaming Mode", use_container_width=True, type="primary"):
            st.session_state.mode = 'live'

    # Initialize mode
    if 'mode' not in st.session_state:
        st.session_state.mode = 'upload'

    st.markdown("---")

    # Display selected mode
    if st.session_state.mode == 'upload':
        video_upload_mode()
    else:
        live_streaming_mode()

    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 🎯 Current Mode: **{st.session_state.mode.upper()}**")

        if st.session_state.mode == 'live':
            st.markdown("---")
            st.markdown("### 📹 Live Mode Guide")
            st.markdown("""
            **Steps:**
            1. 📸 **Capture Floor Frame**
               - Point camera at floor
               - Capture clear image

            2. 🔊 **Generate Audio**
               - AI analyzes floor type
               - Downloads matching sound

            3. ✅ **System Ready**
               - Real-time detection active
               - Walk and hear footsteps!

            **Tips:**
            - Good lighting needed
            - Clear floor view
            - Stand 2-3 meters away
            - Walk naturally
            """)

        st.markdown("---")
        st.markdown("### 🤖 Hybrid Pipeline")
        st.markdown("""
        **Stage 1: YOLO Detection**
        - Detects person in frame
        - Provides bounding box
        - Tracks across frames

        **Stage 2: MediaPipe Pose**
        - Estimates pose on detected region
        - Extracts heel landmarks
        - Higher accuracy & speed

        **Benefits:**
        - ✅ More robust detection
        - ✅ Better occlusion handling
        - ✅ Faster processing
        - ✅ Improved accuracy
        """)

        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.markdown("""
        **Detection Engines:**
        - YOLOv8 (Person Detection)
        - MediaPipe Pose v2 (Pose Estimation)

        **Features:**
        - Dual-stage AI pipeline
        - Person tracking
        - Frame-accurate timing
        - Confidence scoring
        - Real-time live detection
        - Autonomous audio generation
        """)


if __name__ == "__main__":
    main()