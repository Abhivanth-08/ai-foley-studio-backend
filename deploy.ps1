# Quick Deployment Script for Hugging Face Spaces (Windows PowerShell)

Write-Host "🚀 AI Foley Studio Backend - Deployment Helper" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "❌ Error: Not a git repository" -ForegroundColor Red
    Write-Host "Run: git init"
    exit 1
}

Write-Host "📋 Pre-deployment Checklist:" -ForegroundColor Yellow
Write-Host ""

# Check for required files
Write-Host "Checking required files..."
$files = @(
    "Dockerfile",
    "requirements.txt",
    "app.py",
    "link.py",
    "agent.py",
    "custom_wrapper.py",
    "qsec.py",
    "sound_agent.py",
    "yolov8n.pt"
)

$allFilesPresent = $true
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file - MISSING!" -ForegroundColor Red
        $allFilesPresent = $false
    }
}
Write-Host ""

# Check for .env file
if (Test-Path .env) {
    Write-Host "✅ .env file found (don't commit this!)" -ForegroundColor Green
} else {
    Write-Host "⚠️  .env file not found - create one from .env.example" -ForegroundColor Yellow
}
Write-Host ""

# Prompt for Hugging Face username
$hfUsername = Read-Host "Enter your Hugging Face username"
$spaceName = Read-Host "Enter your Space name (default: ai-foley-studio-backend)"
if ([string]::IsNullOrWhiteSpace($spaceName)) {
    $spaceName = "ai-foley-studio-backend"
}

Write-Host ""
Write-Host "📡 Setting up Hugging Face remote..." -ForegroundColor Yellow

try {
    git remote add hf "https://huggingface.co/spaces/$hfUsername/$spaceName" 2>&1 | Out-Null
    Write-Host "✅ Hugging Face remote added successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Remote 'hf' already exists or there was an error" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📝 Checking git status..." -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "Ready to deploy?" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Ensure all files are committed:"
Write-Host "   git add ."
Write-Host "   git commit -m 'Prepare for Hugging Face deployment'"
Write-Host ""
Write-Host "2. Push to GitHub (if not done):"
Write-Host "   git push origin main"
Write-Host ""
Write-Host "3. Push to Hugging Face:"
Write-Host "   git push hf main"
Write-Host ""
Write-Host "4. Set environment variables in HF Space settings:"
Write-Host "   - OPENROUTER_API_KEY"
Write-Host ""
Write-Host "5. Monitor build at:"
Write-Host "   https://huggingface.co/spaces/$hfUsername/$spaceName" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "Do you want to push to Hugging Face now? (y/N)"
if ($confirm -eq "y" -or $confirm -eq "Y" -or $confirm -eq "yes" -or $confirm -eq "Yes") {
    Write-Host ""
    Write-Host "Pushing to Hugging Face..." -ForegroundColor Yellow
    git push hf main
    Write-Host ""
    Write-Host "✅ Deployment initiated!" -ForegroundColor Green
    Write-Host "Check your Space at: https://huggingface.co/spaces/$hfUsername/$spaceName" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Deployment skipped. Run 'git push hf main' when ready." -ForegroundColor Yellow
}
