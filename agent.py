from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from custom_wrapper import OpenRouterChat
from pydantic import BaseModel, Field
from typing import List
import os
import json
import cv2
import base64
from PIL import Image
import io

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class AudioSuggestionOutput(BaseModel):
    audio_suggestions: List[str] = Field(default_factory=list, description="Suggested audio names for footsteps")
    environment_description: str = Field(description="Description of the environment and ground surface")
    reasoning: str = Field(description="Explanation for the audio suggestions")


llm = OpenRouterChat(
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.2-90b-vision-instruct",
    temperature=0.7,
    max_tokens=1024
)

parser = PydanticOutputParser(pydantic_object=AudioSuggestionOutput)


def extract_first_frame(video_path):
    """Extract the first frame from a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError("Cannot read the first frame from video")

        return frame
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return None


def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


prompt = ChatPromptTemplate.from_template("""
You are an expert sound designer and environmental analyst. 
Analyze the provided image and suggest appropriate audio names for footsteps based on the environment, ground surface, and surroundings.

Image Data: {image_data}

Please analyze:
1. The type of ground/surface (concrete, grass, wood, carpet, gravel, etc.)
2. The environment (indoor, outdoor, urban, natural, etc.)
3. Weather conditions if visible (wet, dry, snowy, etc.)
4. Any other relevant factors that would affect footstep sounds
5. Audio suggestion's name must be friendly for a youtube search
6. Name without extensions

Provide 3-5 specific, descriptive audio file name suggestions for footsteps in this environment.
The names should be clear, concise, and follow standard audio naming conventions.

{format_instructions}
""")

chain = (
        {"image_data": RunnablePassthrough(), "format_instructions": lambda x: parser.get_format_instructions()}
        | prompt
        | llm
        | parser
)


def analyze_image_and_suggest_audio(image_base64):
    """Analyze the image and suggest audio names for footsteps"""
    try:
        result = chain.invoke(image_base64)
        return result.dict()
    except Exception as e:
        print("Error during image analysis:", e)
        return None


def process_video_for_footstep_audio(video_path):
    # Extract first frame from video
    print("Extracting first frame from video...")
    first_frame = extract_first_frame(video_path)

    if first_frame is None:
        return {"error": "Failed to extract first frame from video"}

    # Convert image to base64
    print("Converting image to base64...")
    image_base64 = image_to_base64(first_frame)

    if image_base64 is None:
        return {"error": "Failed to convert image to base64"}

    # Analyze image and get audio suggestions
    print("Analyzing image and generating audio suggestions...")
    result = analyze_image_and_suggest_audio(image_base64)

    # Save results
    if result:
        output_file = "found_img1/gemini2.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to {output_file}")

    return result['audio_suggestions'][0]


