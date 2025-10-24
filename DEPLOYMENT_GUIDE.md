# 🚀 Deploying AI Foley Studio Backend to Hugging Face Spaces

## 📋 Prerequisites

1. **Hugging Face Account**: Create account at [huggingface.co](https://huggingface.co)
2. **GitHub Repository**: Your code is already on GitHub
3. **API Keys**: OpenRouter API key for LLM functionality

## 🛠️ Step-by-Step Deployment Guide

### Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `ai-foley-studio-backend` (or your preferred name)
   - **License**: MIT
   - **SDK**: Select **Docker**
   - **Space hardware**: Start with **CPU basic** (free), upgrade if needed
4. Click **"Create Space"**

### Step 2: Connect Your GitHub Repository

You have two options:

#### Option A: Push directly to Hugging Face (Recommended)

1. Add Hugging Face as a remote:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ai-foley-studio-backend
```

2. Push your code:
```bash
git push hf main
```

#### Option B: Link GitHub Repository

1. In your Space settings, go to **"Files and versions"**
2. Use the web interface to upload files or sync with GitHub

### Step 3: Configure Environment Variables (IMPORTANT)

1. In your Space, go to **Settings** → **Variables and secrets**
2. Add the following secret:
   - **Name**: `OPENROUTER_API_KEY`
   - **Value**: Your OpenRouter API key
3. Save the secret

### Step 4: Rename README for Hugging Face

```bash
# Rename README_HF.md to README.md for Hugging Face
mv README_HF.md README_space.md
# Keep your original README.md as is or merge them
```

### Step 5: File Checklist

Make sure these files are in your repository:

✅ **Required Files:**
- [ ] `Dockerfile` - Container configuration
- [ ] `requirements.txt` - Python dependencies
- [ ] `app.py` - Main entry point
- [ ] `link.py` - FastAPI application
- [ ] `reel.py` - Core detection logic
- [ ] `real.py` - Detection pipeline
- [ ] `agent.py` - LLM agent
- [ ] `custom_wrapper.py` - LLM wrapper
- [ ] `qsec.py` - Audio utilities
- [ ] `sound_agent.py` - Audio download (implement this!)
- [ ] `yolov8n.pt` - YOLO model weights
- [ ] `.env.example` - Environment template

⚠️ **Important Notes:**
- The `ffmpeg-7.1-essentials_build/` folder is large - Docker image includes FFmpeg
- Large model files should be under 5GB for free tier

### Step 6: Implement Missing Sound Agent

The `sound_agent.py` file needs to be implemented. Here's what to add:

```python
# Option 1: Use yt-dlp (recommended)
pip install yt-dlp

# Then implement the function to download YouTube audio
```

Or simply use local audio files from the `audio/` folder.

### Step 7: Test Locally with Docker (Optional but Recommended)

```bash
# Build the Docker image
docker build -t ai-foley-studio .

# Run the container
docker run -p 7860:7860 --env-file .env ai-foley-studio
```

Visit `http://localhost:7860` to test.

### Step 8: Deploy and Monitor

1. Once you push to Hugging Face, it will automatically build
2. Monitor the build logs in the **App** tab
3. Wait for "Running" status (may take 5-10 minutes)
4. Your API will be available at: `https://YOUR_USERNAME-ai-foley-studio-backend.hf.space`

## 🔧 Configuration Changes Needed

### 1. Remove Hardcoded Paths

The following files have hardcoded Windows paths that need updating:

**In `link.py`, `real.py`, `reel.py`:**

Current:
```python
r"C:\Users\abhiv\OneDrive\Desktop\agentic ai\SoundFeet\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
```

Should be:
```python
"ffmpeg"  # Docker image has FFmpeg in PATH
```

### 2. Update sound_agent.py

Implement the `main_sound()` function or use local audio files.

### 3. Environment Variables

Create a `.env` file locally (don't commit it!):
```bash
OPENROUTER_API_KEY=your_actual_key_here
```

## 📡 API Usage After Deployment

Your API will be available at:
```
https://YOUR_USERNAME-ai-foley-studio-backend.hf.space/docs
```

Example request:
```bash
curl -X POST "https://YOUR_USERNAME-ai-foley-studio-backend.hf.space/api/upload-video" \
  -F "file=@video.mp4"
```

## 🐛 Troubleshooting

### Build Fails
- Check the build logs in HF Space
- Verify all dependencies in requirements.txt are correct
- Ensure Dockerfile syntax is correct

### API Not Responding
- Check if Space is in "Running" state
- Verify environment variables are set
- Check application logs

### Out of Memory
- Upgrade to better hardware tier (CPU upgrade or GPU)
- Optimize model loading
- Process smaller videos

## 💰 Costs

- **Free Tier**: CPU basic (2 vCPU, 16GB RAM) - sufficient for testing
- **Paid Tiers**: Available if you need more resources
  - CPU upgrade: ~$0.60/hour
  - T4 GPU: ~$0.60/hour

## 🔗 Useful Links

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker SDK Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## ✅ Final Checklist

Before deployment:
- [ ] Implement `sound_agent.py` properly
- [ ] Update FFmpeg paths to use system FFmpeg
- [ ] Test locally with Docker
- [ ] Set up `.env` file with API keys
- [ ] Commit and push to GitHub
- [ ] Create Hugging Face Space
- [ ] Link repository or push to HF
- [ ] Set environment variables in HF Space settings
- [ ] Monitor build and test API

Good luck with your deployment! 🚀
