from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
import requests
import json
import os
import io
from PIL import Image
import base64
import tempfile
import uuid
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import time
import re
import urllib.parse

app = FastAPI(title="House Image Modifier Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Replace with your actual API keys
STABILITY_API_KEY = "sk-SB5H28PO1RD774tJ7vCRAo7PsM2Fbe4wlyuDmOcgBbONe3Fy"
GEMINI_API_KEY = "AIzaSyAiXmO_cO64cFvUUI2vtwj3bBD43SWNEDw"

# Fixed strength parameter for image-to-image operations
FIXED_IMAGE_TO_IMAGE_STRENGTH = 0.65

# Base architectural prompt for text-to-image generation
BASE_ARCHITECTURAL_PROMPT = "Minimalist very small box-shaped square modern house, protruding overhanging flat roof, cantilevered solar panels, clean lines, open and natural surroundings (desert or forest),photorealistic, soft ambient lighting, highly detailed"
DEFAULT_NEGATIVE_PROMPT = (
    "Do not deviate from the prompt and the house should be very small and square shaped. The roof should be protruding only "
)

if not STABILITY_API_KEY or not GEMINI_API_KEY or STABILITY_API_KEY == "YOUR_STABILITY_API_KEY_HERE" or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise ValueError("Please replace the placeholder API keys with your actual API keys")

# Configure Gemini
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel('gemini-1.5-flash')


def save_temp_image_pil(pil_image: Image.Image, filename: str) -> str:
    """Save PIL image to temporary file"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    pil_image.save(temp_path)
    return temp_path

def send_generation_request(host: str, params: dict):
    """Send generation request to Stability AI API for SD3.5"""
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }
    
    # Prepare files and data for multipart form submission
    files = {}
    data = {}
    
    for key, value in params.items():
        if key == "image":
            # Handle image file - must be sent as multipart file
            if isinstance(value, str) and os.path.exists(value):
                files["image"] = ("image", open(value, 'rb'), "image/jpeg")
            else:
                # If it's already a file-like object or bytes
                files["image"] = ("image", value, "image/jpeg")
        else:
            # All other parameters go as form data
            data[key] = str(value)
    
    try:
        # Send as multipart/form-data (files parameter handles this automatically)
        response = requests.post(host, headers=headers, data=data, files=files)
        
        if not response.ok:
            error_detail = f"API Error {response.status_code}: {response.text}"
            raise HTTPException(status_code=response.status_code, detail=error_detail)
        
        return response
    
    finally:
        # Close any opened files
        for file_tuple in files.values():
            if isinstance(file_tuple, tuple) and len(file_tuple) >= 2:
                file_obj = file_tuple[1]
                if hasattr(file_obj, 'close'):
                    file_obj.close()

def stable_diffusion_text_to_image(prompt: str, negative_prompt: str = "", aspect_ratio: str = "1:1", seed: int = 0, output_format: str = "jpeg"):
    """Generate image using SD3.5 Large text-to-image"""
    host = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    # All fields as tuples for multipart/form-data
    files = {
        "prompt": (None, prompt),
        "negative_prompt": (None, negative_prompt),
        "aspect_ratio": (None, aspect_ratio),
        "seed": (None, str(seed)),
        "output_format": (None, output_format),
        "model": (None, "sd3.5-large"),
        "mode": (None, "text-to-image"),
    }
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }
    response = requests.post(host, headers=headers, files=files)
    # Check for NSFW classification
    finish_reason = response.headers.get("finish-reason")
    if finish_reason == 'CONTENT_FILTERED':
        raise HTTPException(status_code=400, detail="Generation failed NSFW classifier")
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.content

def stable_diffusion_image_to_image(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    output_format: str = "jpeg",
    strength: float = 0.65
):
    """Generate image using SD3.5 Large image-to-image with user-defined strength parameter (API doc compliant)"""
    host = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    print(f"[DEBUG] Calling image-to-image API with: image_path={image_path}, prompt={prompt}, negative_prompt={negative_prompt}, seed={seed}, output_format={output_format}, strength={strength}")
    with open(image_path, "rb") as img_file:
        files = {
            "image": ("image.jpg", img_file, "image/jpeg")
        }
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": str(strength),
            "seed": str(seed),
            "output_format": output_format,
            "model": "sd3.5-large",
            "mode": "image-to-image"
        }
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*"
        }
        response = requests.post(host, headers=headers, files=files, data=data)
    # Check for NSFW classification
    finish_reason = response.headers.get("finish-reason")
    if finish_reason == 'CONTENT_FILTERED':
        raise HTTPException(status_code=400, detail="Generation failed NSFW classifier")
    if not response.ok:
        print(f"[DEBUG] image-to-image API error: {response.status_code} {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.text)
    print(f"[DEBUG] image-to-image API call successful, response size: {len(response.content)} bytes")
    return response.content

def replace_background_and_relight(
    subject_image_path: str,
    background_prompt: str = "",
    background_reference_path: str = "",
    foreground_prompt: str = "",
    negative_prompt: str = "",
    preserve_original_subject: float = 0.6,
    original_background_depth: float = 0.5,
    keep_original_background: bool = False,
    light_source_strength: float = 0.3,
    light_reference_path: str = "",
    light_source_direction: str = "none",
    seed: int = 0,
    output_format: str = "jpeg"
) -> bytes:
    """Replace background and relight using Stability AI API - Simplified implementation based on Streamlit logic"""
    print("[DEBUG] replace_background_and_relight called with:")
    print(f"  subject_image_path: {subject_image_path}")
    print(f"  background_prompt: {background_prompt}")
    print(f"  background_reference_path: {background_reference_path}")
    print(f"  foreground_prompt: {foreground_prompt}")
    print(f"  negative_prompt: {negative_prompt}")
    print(f"  preserve_original_subject: {preserve_original_subject}")
    print(f"  original_background_depth: {original_background_depth}")
    print(f"  keep_original_background: {keep_original_background}")
    print(f"  light_source_strength: {light_source_strength}")
    print(f"  light_reference_path: {light_reference_path}")
    print(f"  light_source_direction: {light_source_direction}")
    print(f"  seed: {seed}")
    print(f"  output_format: {output_format}")
    
    # Validate required inputs
    if not background_prompt.strip() and not background_reference_path:
        raise HTTPException(
            status_code=400, 
            detail="Either background_prompt or background_reference_path is required"
        )
    
    host = "https://api.stability.ai/v2beta/stable-image/edit/replace-background-and-relight"
    
    # Prepare headers
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}"
    }
    
    # Prepare files
    files = {}
    with open(subject_image_path, 'rb') as f:
        subject_image_bytes = f.read()
    
    files["subject_image"] = ("image.jpg", subject_image_bytes, "image/jpeg")
    
    if background_reference_path and os.path.exists(background_reference_path):
        with open(background_reference_path, 'rb') as f:
            files["background_reference"] = ("bg_ref.jpg", f.read(), "image/jpeg")
    
    if light_reference_path and os.path.exists(light_reference_path):
        with open(light_reference_path, 'rb') as f:
            files["light_reference"] = ("light_ref.jpg", f.read(), "image/jpeg")
    
    # Prepare parameters
    params = {
        "background_prompt": background_prompt.strip(),
        "foreground_prompt": foreground_prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "preserve_original_subject": preserve_original_subject,
        "original_background_depth": original_background_depth,
        "keep_original_background": keep_original_background,
        "seed": seed,
        "output_format": output_format
    }
    
    # Only add light source parameters if direction is specified
    if light_source_direction and light_source_direction != "none":
        params["light_source_direction"] = light_source_direction
        params["light_source_strength"] = light_source_strength
    
    print(f"[DEBUG] Request URL: {host}")
    print(f"[DEBUG] Request params: {params}")
    print(f"[DEBUG] Request files: {list(files.keys())}")
    
    try:
        # Send initial request
        response = requests.post(host, headers=headers, files=files, data=params)
        
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response headers: {dict(response.headers)}")
        
        if not response.ok:
            print(f"[DEBUG] API error: {response.status_code} {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # Process async response (following Streamlit logic)
        try:
            response_dict = response.json()
            print(f"[DEBUG] Initial response JSON: {response_dict}")
            
            generation_id = response_dict.get("id", None)
            if generation_id is None:
                raise HTTPException(
                    status_code=500,
                    detail="Expected id in response but not found"
                )
            
            print(f"[DEBUG] Generation ID: {generation_id}")
            
            # Poll for results using the same logic as Streamlit
            timeout = 500  # 500 seconds timeout
            start_time = time.time()
            status_code = 202
            
            while status_code == 202:
                print(f"[DEBUG] Polling for results... (Generation ID: {generation_id})")
                
                result_response = requests.get(
                    f"https://api.stability.ai/v2beta/results/{generation_id}",
                    headers={
                        "Accept": "*/*",
                        "Authorization": f"Bearer {STABILITY_API_KEY}"
                    }
                )
                
                print(f"[DEBUG] Poll response status: {result_response.status_code}")
                
                if not result_response.ok:
                    print(f"[DEBUG] Poll error: {result_response.status_code} {result_response.text}")
                    raise HTTPException(
                        status_code=result_response.status_code,
                        detail=f"Polling failed: {result_response.text}"
                    )
                
                status_code = result_response.status_code
                
                if status_code == 200:
                    # Success! Got the result
                    print("[DEBUG] Successfully received result from polling")
                    
                    # Check for NSFW classification
                    finish_reason = result_response.headers.get("finish-reason")
                    if finish_reason == 'CONTENT_FILTERED':
                        raise HTTPException(
                            status_code=400, 
                            detail="Generation failed NSFW classifier"
                        )
                    
                    return result_response.content
                
                # Wait before next poll
                time.sleep(10)
                
                # Check timeout
                if time.time() - start_time > timeout:
                    raise HTTPException(
                        status_code=408,
                        detail=f"Timeout after {timeout} seconds"
                    )
            
            # If we get here, something unexpected happened
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected status code: {status_code}"
            )
            
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response: {response.text}"
            )
        
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

def save_temp_image(image_data: bytes, filename: Optional[str] = None) -> str:
    """Save image data to temporary file"""
    if not filename:
        filename = f"{uuid.uuid4().hex}.png"
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    with open(temp_path, 'wb') as f:
        f.write(image_data)
    
    return temp_path

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def gemini_analyze_request(message: str, has_image: bool = False) -> dict:
    """Use Gemini to analyze the user request and determine the appropriate action"""
    # Custom logic for replace/change and color/colour
    lowered = message.lower()
    # Force replace_background if 'background' is mentioned
    if 'background' in lowered:
        return {
            "action": "replace_background",
            "parameters": {},
            "response": "I will replace the background of the house image as requested."
        }
    if ("replace" in lowered or "change" in lowered):
        if ("colour" in lowered or "color" in lowered):
            # Use search_and_recolor - improved parameter extraction
            # Extract what to recolor and what color to use
            color_term = ""
            select_term = ""
            
            # Try to extract color and object from common patterns
            if "change the color of" in lowered or "change the colour of" in lowered:
                # Extract object after "change the color of"
                start_phrase = "change the color of" if "change the color of" in lowered else "change the colour of"
                start_idx = lowered.find(start_phrase) + len(start_phrase)
                remaining = message[start_idx:].strip()
                if " to " in remaining:
                    select_term = remaining.split(" to ")[0].strip()
                    color_term = remaining.split(" to ", 1)[1].strip()
                else:
                    select_term = remaining.split()[0] if remaining.split() else ""
            elif "make the" in lowered and ("color" in lowered or "colour" in lowered):
                # Extract pattern like "make the roof red" or "make it red"
                start_idx = lowered.find("make the") + len("make the")
                remaining = message[start_idx:].strip()
                words = remaining.split()
                if len(words) >= 2:
                    select_term = words[0]
                    color_term = " ".join(words[1:])
            elif "recolor" in lowered or "recolour" in lowered:
                # Extract recolor patterns
                if " to " in message:
                    parts = message.split(" to ", 1)
                    if len(parts) == 2:
                        color_term = parts[1].strip()
                        # Try to extract object from first part
                        words = parts[0].lower().split()
                        for word in reversed(words):
                            if word not in ["recolor", "recolour", "the", "a", "an"]:
                                select_term = word
                                break
            
            # If we couldn't extract specific terms, use the full message as prompt
            if not color_term:
                color_term = message
            if not select_term:
                # Try to extract common house elements
                house_elements = ["roof", "door", "window", "wall", "siding", "trim", "shutters", "garage", "fence"]
                for element in house_elements:
                    if element in lowered:
                        select_term = element
                        break
                if not select_term:
                    select_term = "house element"
            
            return {
                "action": "search_and_recolor",
                "parameters": {
                    "prompt": color_term,
                    "select_prompt": select_term,
                    "negative_prompt": "",
                    "grow_mask": 3,
                    "seed": 0,
                    "output_format": "webp"
                },
                "response": f"I will change the color of the {select_term} to {color_term} as requested."
            }
        else:
            # Use search_and_replace - improved parameter extraction
            # Extract what to search for and what to replace it with
            search_term = ""
            replacement_description = message
            
            # Try to extract search term from common patterns
            if "replace the" in lowered:
                # Extract what comes after "replace the"
                start_idx = lowered.find("replace the") + len("replace the")
                remaining = message[start_idx:].strip()
                if " with " in remaining:
                    search_term = remaining.split(" with ")[0].strip()
                    replacement_description = remaining.split(" with ", 1)[1].strip()
                elif " to " in remaining:
                    search_term = remaining.split(" to ")[0].strip()
                    replacement_description = remaining.split(" to ", 1)[1].strip()
                else:
                    search_term = remaining.split()[0] if remaining.split() else ""
            elif "change the" in lowered:
                # Extract what comes after "change the"
                start_idx = lowered.find("change the") + len("change the")
                remaining = message[start_idx:].strip()
                if " to " in remaining:
                    search_term = remaining.split(" to ")[0].strip()
                    replacement_description = remaining.split(" to ", 1)[1].strip()
                elif " with " in remaining:
                    search_term = remaining.split(" with ")[0].strip()
                    replacement_description = remaining.split(" with ", 1)[1].strip()
                else:
                    search_term = remaining.split()[0] if remaining.split() else ""
            
            return {
                "action": "search_and_replace",
                "parameters": {
                    "prompt": replacement_description if replacement_description else message,
                    "search_prompt": search_term,
                    "negative_prompt": "",
                    "seed": 0,
                    "output_format": "webp"
                },
                "response": f"I will replace the {search_term if search_term else 'specified element'} in the house image as requested."
            }
    
    # Existing Gemini LLM-based analysis
    system_prompt = """
    You are an AI assistant that analyzes user requests for house/building image operations. 
    Based on the user's message, determine which image editing or generation function to use.
    
    Available functions:
    1. text_to_image - Generate a new house/building image from scratch based on user description
    2. search_and_recolor - Change colors of specific objects (e.g., change roof color, wall color)
    3. search_and_replace - Replace specific objects with new content (e.g., replace windows, doors, roof style)
    4. remove_background - Remove background from image
    5. replace_background - Replace/change the background of the house image with a new environment
    6. structure_api - Minor architectural changes (adding/modifying windows, doors, small structural elements)
    7. image_to_image - Major architectural changes (adding floors, balconies, garages, significant structural modifications)
    8. style_transfer - Apply the style of one image to another image (style transfer between two images)
    
    Decision logic:
    - If user wants to apply the style of one image to another, or mentions 'style transfer', 'apply the style', 'transfer the style', or similar, use 'style_transfer'.
    - If no image is provided AND user wants to create/generate/design a new house/building, use "text_to_image"
    - If user mentions "replace background", "change background", "new background", "different background", "background change", use "replace_background"
    - If image is provided, use appropriate editing function based on the modification requested
    
    Classification rules for existing images:
    - Use "replace_background" for: background changes, environment changes, setting changes, backdrop modifications
    - Use "structure_api" for minor changes: adding/modifying windows, doors, small decorative elements, roof modifications, exterior paint changes
    - Use "image_to_image" for major changes: adding floors/stories, adding balconies, adding garages, adding extensions, completely changing architectural style, major structural additions
    
    For house-specific modifications, consider:
    - Background/environment changes - replace_background
    - Architectural changes (windows, doors, roof modifications) - structure_api
    - Major structural additions (floors, balconies, garages, extensions) - image_to_image
    - Color changes (exterior paint, roof color, trim) - search_and_recolor
    - Landscaping modifications - search_and_replace or image_to_image
    - Adding or removing architectural elements - structure_api for minor, image_to_image for major
    
    IMPORTANT: For image_to_image operations, DO NOT include "strength" parameter in your response. 
    The strength parameter is fixed at 0.65 and cannot be modified.
    
    For replace_background operations, analyze the user's request to extract:
    - background_prompt: Description of the new background/environment (REQUIRED - cannot be empty)
    - foreground_prompt: Description of the house/subject (optional, but helpful for better results)
    - light_source_direction: Extract if user mentions lighting ("left", "right", "above", "below", or "none")
    - preserve_original_subject: How much to preserve the original house (0.0-1.0, default 0.6)
    
    For search_and_replace operations, analyze the user's request to extract:
    - prompt: Description of what to replace the found object with (REQUIRED - detailed description)
    - search_prompt: What to search for and replace (REQUIRED - specific object/element to find)
    - Examples: 
      * "replace the door with a red door" -> search_prompt: "door", prompt: "red door"
      * "change the windows to modern glass windows" -> search_prompt: "windows", prompt: "modern glass windows"
      * "replace roof with tile roof" -> search_prompt: "roof", prompt: "tile roof"
    
    For search_and_recolor operations, analyze the user's request to extract:
    - prompt: Description of the new color/appearance (REQUIRED - e.g., "bright red", "forest green", "navy blue")
    - select_prompt: What object/element to recolor (REQUIRED - e.g., "roof", "door", "walls", "trim")
    - Examples:
      * "make the roof red" -> select_prompt: "roof", prompt: "red"
      * "change the door color to blue" -> select_prompt: "door", prompt: "blue"
      * "paint the walls green" -> select_prompt: "walls", prompt: "green"
    
    CRITICAL for replace_background: 
    - background_prompt MUST NOT be empty or generic. Create a detailed, specific description.
    - If user says "change background to desert", make background_prompt: "vast desert landscape with rolling sand dunes, clear blue sky, sparse desert vegetation, warm golden lighting"
    - Always make foreground_prompt descriptive of the house, like "modern residential house with white walls and dark roof"
    - ALWAYS set output_format to "jpeg" for replace_background operations (matching Streamlit implementation)
    
    CRITICAL for search_and_replace:
    - search_prompt MUST specify exactly what object/element to find and replace
    - prompt MUST describe what to replace it with in detail
    - Both search_prompt and prompt are REQUIRED and cannot be empty
    
    CRITICAL for search_and_recolor:
    - select_prompt MUST specify exactly what object/element to recolor (e.g., "roof", "door", "walls")
    - prompt MUST describe the new color/appearance (e.g., "bright red", "forest green", "navy blue")
    - Both select_prompt and prompt are REQUIRED and cannot be empty
    
    Respond with a JSON object containing:
    {
        "action": "function_name",
        "parameters": {
            "prompt": "description of what to generate/modify",
            "background_prompt": "detailed description of new background (for replace_background - MUST be specific and detailed)",
            "foreground_prompt": "detailed description of the house/subject (for replace_background)",
            "search_prompt": "what to find/select (for search operations) - REQUIRED for search_and_replace",
            "select_prompt": "what to select for recoloring (for search_and_recolor) - REQUIRED",
            "grow_mask": 3,
            "negative_prompt": "what to avoid",
            "preserve_original_subject": 0.6,
            "original_background_depth": 0.5,
            "keep_original_background": false,
            "light_source_strength": 0.3,
            "light_source_direction": "none",
            "seed": 0,
            "output_format": "jpeg",
            "aspect_ratio": "1:1"
        },
        "response": "Friendly explanation of what will be done"
    }
    
    Only include relevant parameters for each function. Never include "strength" parameter for image_to_image.
    For text_to_image, include aspect_ratio parameter (options: "21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21").
    For replace_background, always include detailed background_prompt and foreground_prompt, set output_format to "jpeg".
    For search_and_replace, always include both search_prompt and prompt parameters with specific, detailed values.
    For search_and_recolor, always include both select_prompt and prompt parameters with specific, detailed values.
    """
    
    user_message = f"""
    User request: "{message}"
    Has image: {has_image}
    
    Analyze this request and determine the appropriate action.
    If no image is provided and user wants to create/generate/design a house, use text_to_image.
    If the user mentions replacing, changing, or modifying the background, use replace_background.
    If the user wants to replace specific objects/elements (doors, windows, roof, etc.), use search_and_replace.
    If the user wants to change colors of specific elements, use search_and_recolor.
    If image is provided, determine the appropriate editing function.
    Focus on whether this is a background change (replace_background), color change (search_and_recolor), object replacement (search_and_replace), minor change (structure_api) or major architectural change (image_to_image).
    Remember: Do not include strength parameter for image_to_image operations.
    
    CRITICAL: 
    - If using replace_background, ensure background_prompt is detailed and specific, not generic, and ALWAYS set output_format to "jpeg".
    - If using search_and_replace, ensure both search_prompt and prompt are specific and detailed.
    - If using search_and_recolor, ensure both select_prompt and prompt are specific and detailed.
    
    Example: Instead of "desert", use "vast desert landscape with rolling sand dunes, clear blue sky, sparse cacti and desert plants, warm golden sunlight, distant mountains on horizon"
    """
    
    full_prompt = system_prompt + "\n\n" + user_message
    
    try:
        response = model.generate_content(full_prompt)
        # Extract JSON from response
        response_text = response.text.strip()
        json_text = None
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            # Try to extract JSON from anywhere in the response
            matches = re.findall(r'\{.*\}', response_text, re.DOTALL)
            if matches:
                json_text = matches[0]
            else:
                json_text = response_text
        try:
            result = json.loads(json_text)
        except Exception as e:
            print("[DEBUG] Failed to parse Gemini JSON:", str(e), "\nRaw response:", response_text)
            raise e
        
        # Ensure strength parameter is removed from image_to_image operations
        if result.get("action") == "image_to_image" and "parameters" in result:
            result["parameters"].pop("strength", None)
        
        # Validate replace_background parameters
        if result.get("action") == "replace_background" and "parameters" in result:
            params = result["parameters"]
            # Ensure background_prompt is not empty or too generic
            bg_prompt = params.get("background_prompt", "").strip()
            if not bg_prompt or len(bg_prompt) < 20:
                # Generate a more detailed background prompt
                params["background_prompt"] = "detailed outdoor environment with natural lighting, realistic landscape setting, clear sky, appropriate atmospheric conditions"
            
            # Ensure foreground_prompt describes the house
            if not params.get("foreground_prompt", "").strip():
                params["foreground_prompt"] = "the house from the original image, maintaining its architectural style and proportions"
            
            # Force JPEG output for replace_background (matching Streamlit)
            params["output_format"] = "jpeg"
        
        # Validate search_and_replace parameters
        if result.get("action") == "search_and_replace" and "parameters" in result:
            params = result["parameters"]
            # Ensure search_prompt is not empty
            if not params.get("search_prompt", "").strip():
                params["search_prompt"] = "object to replace"
            
            # Ensure prompt is not empty
            if not params.get("prompt", "").strip():
                params["prompt"] = "replacement object or element"
        
        # Validate search_and_recolor parameters
        if result.get("action") == "search_and_recolor" and "parameters" in result:
            params = result["parameters"]
            # Ensure select_prompt is not empty
            if not params.get("select_prompt", "").strip():
                params["select_prompt"] = "house element"
            
            # Ensure prompt is not empty
            if not params.get("prompt", "").strip():
                params["prompt"] = "new color"
        
        # If Gemini says style_transfer, include show_style_transfer_button
        if result.get("action") == "style_transfer":
            result["show_style_transfer_button"] = True
        
        return result
    except Exception as e:
        print("[DEBUG] Gemini analyze_request failed:", str(e))
        return {
            "action": "clarification_needed",
            "response": "Sorry, I couldn't understand your request. Could you please rephrase or be more specific?",
            "parameters": {}
        }

def save_image_to_static(image_data: bytes, filename: Optional[str] = None) -> str:
    static_dir = os.path.join(os.getcwd(), "static", "images")
    os.makedirs(static_dir, exist_ok=True)
    if not filename:
        filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(static_dir, filename)
    with open(file_path, "wb") as f:
        f.write(image_data)
    print(f"[DEBUG] Image saved to: {file_path}")
    return file_path

def search_and_replace(image_path: str, prompt: str, search_prompt: str, negative_prompt: str = "", seed: int = 0, output_format: str = "webp"):
    """Search and replace using Stability AI API (API doc compliant)"""
    host = "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace"
    with open(image_path, "rb") as img_file:
        files = {
            "image": ("image.jpg", img_file, "image/jpeg")
        }
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "search_prompt": search_prompt,
            "seed": str(seed),
            "mode": "search",
            "output_format": output_format
        }
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*"
        }
        response = requests.post(host, headers=headers, files=files, data=data)
    finish_reason = response.headers.get("finish-reason")
    if finish_reason == 'CONTENT_FILTERED':
        raise HTTPException(status_code=400, detail="Generation failed NSFW classifier")
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.content

def search_and_recolor(image_path: str, prompt: str, select_prompt: str, negative_prompt: str = "", grow_mask: int = 3, seed: int = 0, output_format: str = "webp"):
    """Search and recolor using Stability AI API (API doc compliant, with Gemini prompt cleaning and debug output)"""
    # Use Gemini to clean and optimize the prompt and select_prompt
    gemini_prompt = (
        "You are an expert at preparing prompts for an AI image recoloring API. "
        "Given a user request, generate the most concise and effective 'select_prompt' (the object to recolor, e.g. 'roof', 'door', 'window') "
        "and 'prompt' (the new color/appearance, e.g. 'red roof', 'blue door', 'green window'). "
        "Respond in JSON: { 'select_prompt': ..., 'prompt': ... }. "
        f"User select_prompt: '{select_prompt}', user prompt: '{prompt}'"
    )
    try:
        gemini_response = model.generate_content(gemini_prompt)
        response_text = gemini_response.text.strip()
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        gemini_result = json.loads(json_text)
        select_prompt = gemini_result.get('select_prompt', select_prompt)
        prompt = gemini_result.get('prompt', prompt)
    except Exception as e:
        print('[DEBUG] Gemini prompt cleaning failed:', str(e))
    host = "https://api.stability.ai/v2beta/stable-image/edit/search-and-recolor"
    with open(image_path, "rb") as img_file:
        files = {
            "image": ("image.jpg", img_file, "image/jpeg")
        }
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "select_prompt": select_prompt,
            "grow_mask": str(grow_mask),
            "seed": str(seed),
            "mode": "search",
            "output_format": output_format
        }
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*"
        }
        print("[DEBUG] search_and_recolor API endpoint:", host)
        print("[DEBUG] Data sent to API:", data)
        print("[DEBUG] Files sent to API:", list(files.keys()))
        response = requests.post(host, headers=headers, files=files, data=data)
    print("[DEBUG] Response status code:", response.status_code)
    print("[DEBUG] Response headers:", dict(response.headers))
    finish_reason = response.headers.get("finish-reason")
    if finish_reason == 'CONTENT_FILTERED':
        raise HTTPException(status_code=400, detail="Generation failed NSFW classifier")
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.content

def call_structure_api(image_path: str, prompt: str, negative_prompt: str = "", control_strength: float = 0.7, seed: int = 0, output_format: str = "jpeg") -> bytes:
    """Call the Stability AI control/structure API for minor architectural changes."""
    host = "https://api.stability.ai/v2beta/stable-image/control/structure"
    
    params = {
        "image": image_path,
        "control_strength": control_strength,
        "seed": seed,
        "output_format": output_format,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    
    try:
        response = send_generation_request(host, params)
        
        # Check for NSFW classification
        finish_reason = response.headers.get("finish-reason")
        if finish_reason == 'CONTENT_FILTERED':
            raise HTTPException(status_code=400, detail="Generation failed NSFW classifier")
        
        return response.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure API failed: {str(e)}")

def build_final_prompt(user_prompt: str) -> str:
    # If user prompt already specifies storey/floor, don't add extra info
    storey_keywords = [
        "one storey", "single floor", "ground floor only", "no upper floor", "1 storey", "1 floor", "single storey", "only one floor"
    ]
    user_prompt_lower = user_prompt.lower()
    if any(word in user_prompt_lower for word in storey_keywords):
        return f"{BASE_ARCHITECTURAL_PROMPT} {user_prompt}"
    else:
        return f"{BASE_ARCHITECTURAL_PROMPT} {user_prompt}"

def is_local_house_image(url):
    decoded_url = urllib.parse.unquote(url)
    return (
        decoded_url.startswith("/House Images/")
        or re.match(r"http://(localhost|127\\.0\\.0\\.1):8000/House Images/", decoded_url)
    )