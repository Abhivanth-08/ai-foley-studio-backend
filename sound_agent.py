import os
import yt_dlp
import requests
from bs4 import BeautifulSoup
import re

# Set the path to your FFmpeg executable
FFMPEG_PATH = r"C:\Users\abhiv\OneDrive\Desktop\agentic ai\SoundFeet\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"

def create_audio_folder():
    """Create audio folder if it doesn't exist"""
    if not os.path.exists("audio"):
        os.makedirs("audio")
    return "audio"


def check_ffmpeg():
    """Check if FFmpeg is available at the specified path"""
    if not os.path.exists(FFMPEG_PATH):
        print(f"❌ FFmpeg not found at: {FFMPEG_PATH}")
        print("Please check the path and make sure FFmpeg is installed.")
        return False
    print(f"✅ FFmpeg found at: {FFMPEG_PATH}")
    return True


def search_and_download_audio(audio_name):
    """Search and download audio using yt-dlp's built-in search"""
    audio_folder = create_audio_folder()
    sanitized_name = sanitize_filename(audio_name)

    # Configure yt-dlp with FFmpeg path
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{audio_folder}/{sanitized_name}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': os.path.dirname(FFMPEG_PATH),
        'default_search': 'ytsearch',  # Use YouTube search
        'noplaylist': True,  # Download only single video, not playlist
    }

    try:
        print(f"🔍 Searching for '{audio_name}' on YouTube...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search and download the first result
            search_query = f"{audio_name} audio"
            ydl.download([search_query])

        # Check if file was created
        mp3_file = os.path.join(audio_folder, f"{sanitized_name}.mp3")
        if os.path.exists(mp3_file):
            file_size = os.path.getsize(mp3_file) / (1024 * 1024)  # Size in MB
            print(f"✅ Audio '{sanitized_name}' downloaded successfully! ({file_size:.2f} MB)")
            return ydl_opts['outtmpl']
        else:
            print("❌ Downloaded file not found.")
            return False

    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Download error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def search_youtube_improved(audio_name):
    """Alternative search method with better headers"""
    search_query = f"{audio_name} audio"
    url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Extract video IDs using regex from the page source
        video_ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', response.text)

        # Remove duplicates and create full URLs
        video_links = []
        for video_id in video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"
            if url not in video_links:
                video_links.append(url)

        return video_links[:5]  # Return top 5 results

    except Exception as e:
        print(f"❌ Error searching YouTube: {e}")
        return []


def sanitize_filename(name):
    """Remove invalid characters from filename"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '')
    return name.strip()


def main_sound(audio_name):
    print("🎵 Audio Downloader")
    print("=" * 40)

    # Check FFmpeg availability first
    if not check_ffmpeg():
        return None
    if not audio_name:
        print("❌ Please enter a valid audio name.")
        return None

    # Try the direct download method first (more reliable)
    print("\n🔄 Trying direct download method...")
    file_path = search_and_download_audio(audio_name)
    if file_path:
        print(f"🎉 Success! Audio saved as '{sanitize_filename(audio_name)}.mp3'")
        return file_path
    else:
        print("\n🔄 Direct method failed, trying alternative search...")

        # Try alternative search method
        video_urls = search_youtube_improved(audio_name)

        if not video_urls:
            print("❌ No audio found. Please try a different name.")
            print(
                "💡 Try more specific terms like: 'city street sounds', 'footsteps on pavement', 'urban ambient noise'")
            return None

        print(f"📥 Found {len(video_urls)} results. Downloading the first one...")

        # Download using the traditional method
        file_path = download_audio_direct(audio_name, video_urls[0])
        if file_path:
            print(f"🎉 Audio saved in 'audio' folder!")
            return file_path
        else:
            print("❌ All download methods failed.")
            return None


def download_audio_direct(audio_name, url):
    """Direct download method for specific URLs"""
    audio_folder = create_audio_folder()
    sanitized_name = sanitize_filename(audio_name)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{audio_folder}/{sanitized_name}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': os.path.dirname(FFMPEG_PATH),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return ydl_opts['outtmpl']
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

