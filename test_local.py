#!/usr/bin/env python3
"""
Local Test Script - Test your API before deployment
Run this to verify everything works locally
"""

import os
import sys
import time
import requests
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_environment():
    """Check if environment is set up correctly"""
    print_header("Checking Environment")
    
    # Check .env file
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("❌ .env file not found - create one from .env.example")
        return False
    
    # Check if OPENROUTER_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key and api_key != 'your_openrouter_api_key_here':
        print("✅ OPENROUTER_API_KEY is set")
    else:
        print("❌ OPENROUTER_API_KEY not set properly in .env")
        return False
    
    # Check required files
    required_files = [
        'link.py', 'agent.py', 'custom_wrapper.py', 
        'qsec.py', 'reel.py', 'sound_agent.py', 'yolov8n.pt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            return False
    
    return True

def test_imports():
    """Test if all imports work"""
    print_header("Testing Imports")
    
    try:
        print("Testing FastAPI imports...")
        from fastapi import FastAPI
        print("✅ FastAPI")
        
        print("Testing CV libraries...")
        import cv2
        print("✅ OpenCV")
        
        import mediapipe as mp
        print("✅ MediaPipe")
        
        from ultralytics import YOLO
        print("✅ YOLO")
        
        print("Testing audio libraries...")
        import librosa
        print("✅ Librosa")
        
        import soundfile as sf
        print("✅ SoundFile")
        
        print("Testing LangChain...")
        from langchain_core.prompts import ChatPromptTemplate
        print("✅ LangChain")
        
        print("Testing utilities...")
        import numpy as np
        import pandas as pd
        from PIL import Image
        print("✅ NumPy, Pandas, Pillow")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_api_endpoints(base_url="http://localhost:7860"):
    """Test API endpoints"""
    print_header("Testing API Endpoints")
    
    print(f"Base URL: {base_url}")
    print("Waiting for server to be ready...")
    
    # Wait for server to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/")
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            if i < max_retries - 1:
                print(f"   Waiting... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("❌ Server not responding")
                return False
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GET / - {data}")
        else:
            print(f"❌ GET / failed - Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing root endpoint: {e}")
        return False
    
    # Test docs endpoint
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print(f"✅ GET /docs - API documentation available")
            print(f"   Visit: {base_url}/docs")
        else:
            print(f"⚠️  GET /docs - Status: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Error testing docs endpoint: {e}")
    
    print("\n✅ Basic API tests passed!")
    print(f"\n🌐 Open your browser to: {base_url}/docs")
    return True

def main():
    print("\n" + "🧪 AI Foley Studio - Local Testing".center(60))
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Step 2: Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install requirements.")
        sys.exit(1)
    
    # Step 3: Instructions for running server
    print_header("Starting Server")
    print("To start the server, run in another terminal:")
    print("  uvicorn link:app --host 0.0.0.0 --port 7860 --reload")
    print("\nOr use:")
    print("  python app.py")
    
    input("\nPress Enter once the server is running...")
    
    # Step 4: Test API endpoints
    if not test_api_endpoints():
        print("\n❌ API test failed.")
        sys.exit(1)
    
    # Success!
    print_header("🎉 All Tests Passed!")
    print("Your application is ready for deployment!")
    print("\nNext steps:")
    print("  1. Test uploading a video through the API docs")
    print("  2. Verify video processing works")
    print("  3. Deploy to Hugging Face Spaces")
    print("\nRefer to DEPLOYMENT_GUIDE.md for deployment instructions.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
