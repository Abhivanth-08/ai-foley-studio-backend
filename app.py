"""
Main application entry point for Hugging Face Spaces deployment
"""
from link2 import app

# This file serves as the main entry point
# The app is imported from link.py which contains the FastAPI application

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
