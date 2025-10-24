#!/bin/bash
# Quick Deployment Script for Hugging Face Spaces

echo "🚀 AI Foley Studio Backend - Deployment Helper"
echo "=============================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    echo "Run: git init"
    exit 1
fi

echo "📋 Pre-deployment Checklist:"
echo ""

# Check for required files
echo "Checking required files..."
files=(
    "Dockerfile"
    "requirements.txt"
    "app.py"
    "link.py"
    "agent.py"
    "custom_wrapper.py"
    "qsec.py"
    "sound_agent.py"
    "yolov8n.pt"
)

all_files_present=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING!"
        all_files_present=false
    fi
done
echo ""

# Check for .env file
if [ -f .env ]; then
    echo "✅ .env file found (don't commit this!)"
else
    echo "⚠️  .env file not found - create one from .env.example"
fi
echo ""

# Prompt for Hugging Face username
read -p "Enter your Hugging Face username: " hf_username
read -p "Enter your Space name (default: ai-foley-studio-backend): " space_name
space_name=${space_name:-ai-foley-studio-backend}

echo ""
echo "📡 Setting up Hugging Face remote..."
git remote add hf "https://huggingface.co/spaces/${hf_username}/${space_name}" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Hugging Face remote added successfully"
else
    echo "⚠️  Remote 'hf' already exists or there was an error"
fi

echo ""
echo "📝 Checking git status..."
git status --short

echo ""
echo "Ready to deploy?"
echo ""
echo "Next steps:"
echo "1. Ensure all files are committed:"
echo "   git add ."
echo "   git commit -m 'Prepare for Hugging Face deployment'"
echo ""
echo "2. Push to GitHub (if not done):"
echo "   git push origin main"
echo ""
echo "3. Push to Hugging Face:"
echo "   git push hf main"
echo ""
echo "4. Set environment variables in HF Space settings:"
echo "   - OPENROUTER_API_KEY"
echo ""
echo "5. Monitor build at:"
echo "   https://huggingface.co/spaces/${hf_username}/${space_name}"
echo ""

read -p "Do you want to push to Hugging Face now? (y/N): " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo ""
    echo "Pushing to Hugging Face..."
    git push hf main
    echo ""
    echo "✅ Deployment initiated!"
    echo "Check your Space at: https://huggingface.co/spaces/${hf_username}/${space_name}"
else
    echo ""
    echo "Deployment skipped. Run 'git push hf main' when ready."
fi
