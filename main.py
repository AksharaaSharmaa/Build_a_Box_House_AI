from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
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
from image_edit_functions import *

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
BASE_ARCHITECTURAL_PROMPT = (
    "Minimalist very small box-shaped square modern house, "
    "flat roofs with wide overhangs, creating bold shadows, "
    "the house should be square and small and with outwards protruding roof, "
    "cantilevered solar panels, clean lines, open and natural surroundings (desert or forest), "
    "The whole house should look like one box only regardless of the number of floors."
)
DEFAULT_NEGATIVE_PROMPT = (
    "Do not deviate from the prompt."
)

if not STABILITY_API_KEY or not GEMINI_API_KEY or STABILITY_API_KEY == "YOUR_STABILITY_API_KEY_HERE" or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise ValueError("Please replace the placeholder API keys with your actual API keys")

# Configure Gemini
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel('gemini-1.5-flash')

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None
    image_base64: Optional[Union[str, List[str]]] = None
    strength: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    action: str
    image_url: Optional[str] = None
    parameters: Optional[dict] = None
    needs_strength: Optional[bool] = None
    show_style_transfer_button: Optional[bool] = None

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
STABILITY_API_KEY = "sk-SB5H28PO1RD774tJ7vCRAo7PsM2Fbe4wlyuDmOcgBbONe3Fy"
INPAINT_API_URL = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
OUTPAINT_API_URL = "https://api.stability.ai/v2beta/stable-image/edit/outpaint"

# Store session data (in production, use Redis or database)
sessions = {}

def send_outpaint_request(image_data, left=0, right=0, up=200, down=0, prompt="", creativity=0.5, seed=0, output_format="webp"):
    """Send outpainting request to Stability AI API"""
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }
    
    files = {
        "image": ("image.png", image_data, "image/png")
    }
    
    data = {
        "left": left,
        "right": right,
        "up": up,
        "down": down,
        "prompt": prompt,
        "creativity": creativity,
        "seed": seed,
        "output_format": output_format
    }
    
    response = requests.post(OUTPAINT_API_URL, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        return response.content, response.headers.get("seed", "unknown")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

def send_mask_request(image_data, mask_data, prompt, negative_prompt="", seed=0, output_format="webp"):
    """Send inpainting request to Stability AI API"""
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }
    
    files = {
        "image": ("image.png", image_data, "image/png"),
        "mask": ("mask.png", mask_data, "image/png")
    }
    
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "output_format": output_format,
        "mode": "mask"
    }
    
    response = requests.post(INPAINT_API_URL, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        return response.content, response.headers.get("seed", "unknown")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

def create_mask_from_canvas_data(canvas_data, image_size):
    """Create mask from canvas drawing data"""
    # Decode base64 canvas data
    canvas_data = canvas_data.replace("data:image/png;base64,", "")
    canvas_bytes = base64.b64decode(canvas_data)
    
    # Convert to PIL Image
    canvas_img = Image.open(io.BytesIO(canvas_bytes))
    
    # Resize to match original image size
    canvas_img = canvas_img.resize(image_size, Image.Resampling.LANCZOS)
    
    # Convert to grayscale and create mask
    # Non-transparent areas become white (masked areas)
    mask = Image.new('L', image_size, 0)
    
    if canvas_img.mode == 'RGBA':
        # Get alpha channel
        alpha = canvas_img.split()[-1]
        mask_array = np.array(alpha)
        # Convert non-transparent areas to white
        mask_array[mask_array > 0] = 255
        mask = Image.fromarray(mask_array, 'L')
    else:
        # Convert to RGB and find drawn areas
        canvas_rgb = canvas_img.convert('RGB')
        pixels = np.array(canvas_rgb)
        drawn_areas = np.any(pixels != [255, 255, 255], axis=2)
        
        mask_array = np.zeros(image_size[::-1], dtype=np.uint8)
        mask_array[drawn_areas] = 255
        mask = Image.fromarray(mask_array, 'L')
    
    return mask

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main HTML page"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload")
async def upload_image(file: List[UploadFile] = File(...)):
    """Upload house image endpoint (multiple files supported)"""
    images_base64 = []
    filenames = []
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    for f in file:
        if not f.content_type or not f.content_type.startswith('image/'):
            continue
        image_data = await f.read()
        images_base64.append(base64.b64encode(image_data).decode())
        filenames.append(f.filename)
        # Store image data in session for each file (if you want to support multiple, you can extend this logic)
        sessions[session_id] = {
            "original_image": image_data,
            "image_size": Image.open(io.BytesIO(image_data)).size,
            "outpainted_image": None
        }
    if not images_base64:
        raise HTTPException(status_code=400, detail="No valid image files uploaded")
    return {"images_base64": images_base64, "filenames": filenames, "session_id": session_id}

@app.post("/outpaint")
async def outpaint_image(
    image: UploadFile = File(...),
    extend_height: int = Form(500),
    creativity: float = Form(0.1),
    outpaint_prompt: str = Form("natural sky, clouds, same architectural style"),
    seed: int = Form(0),
    output_format: str = Form("webp")
):
    """Extend canvas upwards (stateless, no session)"""
    try:
        print("[OUTPAINT] Received request")
        print(f"[OUTPAINT] image.filename: {image.filename}, content_type: {image.content_type}")
        print(f"[OUTPAINT] extend_height: {extend_height}, creativity: {creativity}, outpaint_prompt: {outpaint_prompt}, seed: {seed}, output_format: {output_format}")
        image_data = await image.read()
        print(f"[OUTPAINT] image_data size: {len(image_data)} bytes")
        outpaint_result, outpaint_seed = send_outpaint_request(
            image_data,
            left=0,
            right=0,
            up=extend_height,
            down=0,
            prompt=outpaint_prompt,
            creativity=creativity,
            seed=seed,
            output_format=output_format
        )
        outpainted_pil = Image.open(io.BytesIO(outpaint_result))
        outpaint_base64 = base64.b64encode(outpaint_result).decode()
        print(f"[OUTPAINT] outpainted image size: {outpainted_pil.size}, seed: {outpaint_seed}")
        return {
            "success": True,
            "image_data": f"data:image/png;base64,{outpaint_base64}",
            "image_size": outpainted_pil.size,
            "seed": outpaint_seed
        }
    except Exception as e:
        print(f"[OUTPAINT][ERROR] {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/inpaint")
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    seed: int = Form(0),
    output_format: str = Form("webp"),
    mode: str = Form("Manual Mode")
):
    """Perform inpainting on the image (stateless, no session)"""
    try:
        print("[INPAINT] Received request")
        print(f"[INPAINT] image.filename: {image.filename}, content_type: {image.content_type}")
        print(f"[INPAINT] mask.filename: {mask.filename}, content_type: {mask.content_type}")
        print(f"[INPAINT] prompt: {prompt}, negative_prompt: {negative_prompt}, seed: {seed}, output_format: {output_format}, mode: {mode}")
        image_data = await image.read()
        mask_data = await mask.read()
        print(f"[INPAINT] image_data size: {len(image_data)} bytes, mask_data size: {len(mask_data)} bytes")
        result_image, result_seed = send_mask_request(
            image_data, mask_data, prompt, negative_prompt, seed, output_format
        )
        result_base64 = base64.b64encode(result_image).decode()
        mask_base64 = base64.b64encode(mask_data).decode()
        print(f"[INPAINT] result image size: {len(result_image)} bytes, seed: {result_seed}")
        return {
            "success": True,
            "image_data": f"data:image/{output_format};base64,{result_base64}",
            "mask_data": f"data:image/png;base64,{mask_base64}",
            "seed": result_seed,
            "filename": f"inpainted_{result_seed}.{output_format}"
        }
    except Exception as e:
        print(f"[INPAINT][ERROR] {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/download/{session_id}")
async def download_result(session_id: str):
    """Download the generated image"""
    if session_id not in sessions or "result_image" not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="No result image found")
    
    result_data = sessions[session_id]["result_image"]
    return StreamingResponse(
        io.BytesIO(result_data),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=inpainted_result.png"}
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes house modification requests and generation"""
    try:
        print("[DEBUG] /chat called")
        print(f"[DEBUG] Incoming request: message={request.message!r}, image_url={request.image_url!r}, image_base64={'set' if request.image_base64 else 'None'}, strength={request.strength}")
        # Check if image is provided
        has_image = bool(request.image_url or request.image_base64)
        print(f"[DEBUG] has_image: {has_image}")
        if request.image_url:
            print(f"[DEBUG] image_url: {request.image_url}")
        if request.image_base64:
            print(f"[DEBUG] image_base64: Provided")
        # Analyze request with Gemini
        print(f"[DEBUG] Calling gemini_analyze_request with message: {request.message!r}, has_image: {has_image}")
        analysis = gemini_analyze_request(request.message, has_image)
        print(f"[DEBUG] Gemini analysis result: {json.dumps(analysis, indent=2)}")
        # If Gemini says style_transfer, return response with show_style_transfer_button even if no image is uploaded
        if analysis["action"] == "style_transfer":
            print("[DEBUG] Gemini selected action: style_transfer (showing style transfer button)")
            return ChatResponse(
                response=analysis.get("response", "You can use the style transfer feature for this request."),
                action="style_transfer",
                show_style_transfer_button=True
            )
        # For all other actions, image is required
        if not has_image:
            print(f"[DEBUG] No image provided, Gemini action: {analysis['action']}")
            if analysis["action"] == "text_to_image":
                # Generate image using SD3.5 Large text-to-image
                params = analysis.get("parameters", {})
                user_prompt = params.get("prompt", request.message)
                merged_prompt = build_final_prompt(user_prompt)
                negative_prompt = params.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
                try:
                    result_image = stable_diffusion_text_to_image(
                        merged_prompt,
                        negative_prompt,
                        params.get("aspect_ratio", "1:1"),
                        params.get("seed", 0),
                        params.get("output_format", "jpeg")
                    )
                    # Save result image to static/images/
                    result_path = save_image_to_static(result_image)
                    filename = os.path.basename(result_path)
                    image_url = f"/static/images/{filename}"
                    params["merged_prompt"] = merged_prompt
                    return ChatResponse(
                        response=analysis["response"],
                        action="text_to_image",
                        image_url=image_url,
                        parameters=params
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
            else:
                print("[DEBUG] Returning: Please provide a house image to modify, or describe a house you'd like me to generate for you.")
                return ChatResponse(
                    response="Please provide a house image to modify, or describe a house you'd like me to generate for you.",
                    action="no_image",
                    parameters={}
                )
        
        # Force structure_api if prompt contains 'change the architectural style to'
        if "change the architectural style to" in request.message.lower():
            print("[DEBUG] Forcing method: structure_api due to explicit prompt phrase.")
            analysis["action"] = "structure_api"
        
        # Download/process image
        image_data = None
        if request.image_url:
            decoded_url = urllib.parse.unquote(request.image_url)
            if is_local_house_image(request.image_url):
                # Convert to local path
                if decoded_url.startswith("/House Images/"):
                    local_path = os.path.join(os.getcwd(), decoded_url.lstrip("/"))
                else:
                    # Remove http://localhost:8000 or http://127.0.0.1:8000
                    local_path = re.sub(r"^http://(localhost|127\\.0\\.0\\.1):8000/", "", decoded_url)
                    local_path = os.path.join(os.getcwd(), local_path.lstrip("/"))
                print(f"[DEBUG] Loading local image from: {local_path}")
                if not os.path.exists(local_path):
                    print(f"[DEBUG] ERROR: Local image file not found: {local_path}")
                    raise HTTPException(status_code=400, detail="Could not find local image file")
                with open(local_path, "rb") as f:
                    image_data = f.read()
            else:
                # Remote or static image, download via HTTP
                print(f"[DEBUG] Downloading image from URL: {request.image_url}")
                img_response = requests.get(request.image_url)
                if not img_response.ok:
                    print(f"[DEBUG] ERROR: Could not download image from URL: {request.image_url}")
                    raise HTTPException(status_code=400, detail="Could not download image from URL")
                image_data = img_response.content
        elif request.image_base64:
            print("[DEBUG] Decoding image from base64")
            if isinstance(request.image_base64, list):
                image_data = base64.b64decode(request.image_base64[0])
            else:
                image_data = base64.b64decode(request.image_base64)
        
        # Save image to temporary file
        if image_data is None:
            print("[DEBUG] ERROR: No image data found to process.")
            raise HTTPException(status_code=400, detail="No image data found to process.")
        temp_image_path = save_temp_image(image_data)
        print(f"[DEBUG] Saved temp image to: {temp_image_path}")
        
        # Execute the appropriate action
        action = analysis["action"]
        params = analysis.get("parameters", {})
        
        try:
            print(f"[DEBUG] Starting action: {action}")
            if action == "replace_background":
                print("[DEBUG] Calling replace_background_and_relight...")
                # If background_prompt is missing, use Gemini to generate it
                background_prompt = params.get("background_prompt", "").strip()
                if not background_prompt:
                    # Use Gemini to generate a detailed background prompt based on the user's message
                    gemini_bg_system_prompt = (
                        "You are an expert prompt engineer. Given a user request for a new background for a house image, "
                        "generate a vivid, detailed, and specific background description suitable for an AI image generation model. "
                        "Do NOT mention the house or subject, only describe the background environment. "
                        "Be creative and descriptive. Example: 'a lush green forest with tall pine trees, dappled sunlight, and a misty morning atmosphere'. "
                        "User request: '" + request.message + "'\nRespond ONLY with the background description, nothing else."
                    )
                    try:
                        gemini_response = model.generate_content(gemini_bg_system_prompt)
                        response_text = gemini_response.text.strip()
                        # Use the Gemini response directly as the background prompt
                        background_prompt = response_text
                    except Exception as e:
                        background_prompt = request.message.strip()
                    if not background_prompt:
                        background_prompt = "detailed outdoor environment with natural lighting, realistic landscape setting, clear sky, appropriate atmospheric conditions"
                    params["background_prompt"] = background_prompt
                # Replace background and relight - FORCE PNG OUTPUT
                result_image = replace_background_and_relight(
                    subject_image_path=temp_image_path,
                    background_prompt=background_prompt,
                    foreground_prompt=params.get("foreground_prompt", ""),
                    negative_prompt=params.get("negative_prompt", ""),
                    preserve_original_subject=params.get("preserve_original_subject", 0.6),
                    original_background_depth=params.get("original_background_depth", 0.5),
                    keep_original_background=params.get("keep_original_background", False),
                    light_source_strength=params.get("light_source_strength", 0.3),
                    light_source_direction=params.get("light_source_direction", "none"),
                    seed=params.get("seed", 0),
                    output_format="png"  # Force PNG output
                )
                print("[DEBUG] replace_background_and_relight finished.")
                
            elif action == "image_to_image":
                print("[DEBUG] Calling stable_diffusion_image_to_image...")
                # If strength is not provided, prompt frontend to show slider
                strength = request.strength if request.strength is not None else params.get("strength")
                if strength is None:
                    print("[DEBUG] No strength provided, returning needs_strength=True")
                    # Ask frontend to show slider (do not generate image yet)
                    return ChatResponse(
                        response="Please select the strength for the image-to-image transformation.",
                        action="image_to_image",
                        parameters=params,
                        needs_strength=True
                    )
                # Major architectural changes using SD3.5 Large with user-selected strength
                result_image = stable_diffusion_image_to_image(
                    temp_image_path,
                    params.get("prompt", request.message),
                    params.get("negative_prompt", ""),
                    params.get("seed", 0),
                    params.get("output_format", "jpeg"),
                    strength=strength
                )
                params["strength"] = strength
                print("[DEBUG] stable_diffusion_image_to_image finished.")
                
            elif action == "structure_api":
                print("[DEBUG] Calling call_structure_api...")
                # Minor architectural changes using structure API
                result_image = call_structure_api(
                    temp_image_path,
                    params.get("prompt", request.message),
                    params.get("negative_prompt", ""),
                    control_strength=params.get("control_strength", 0.7),
                    seed=params.get("seed", 0),
                    output_format=params.get("output_format", "jpeg")
                )
                print("[DEBUG] call_structure_api finished.")
                
            elif action == "search_and_recolor":
                print("[DEBUG] Calling search_and_recolor...")
                result_image = search_and_recolor(
                    temp_image_path,
                    params.get("prompt", ""),
                    params.get("select_prompt", ""),
                    params.get("negative_prompt", ""),
                    params.get("grow_mask", 3),
                    params.get("seed", 0)
                )
                print("[DEBUG] search_and_recolor finished.")
                
            elif action == "search_and_replace":
                print("[DEBUG] Calling search_and_replace...")
                result_image = search_and_replace(
                    temp_image_path,
                    params.get("prompt", ""),
                    params.get("search_prompt", ""),
                    params.get("negative_prompt", ""),
                    params.get("seed", 0)
                )
                print("[DEBUG] search_and_replace finished.")
            
            elif action == "text_to_image":
                print("[DEBUG] Calling stable_diffusion_text_to_image (with image present)...")
                params = analysis.get("parameters", {})
                user_prompt = params.get("prompt", request.message)
                merged_prompt = build_final_prompt(user_prompt)
                negative_prompt = params.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
                try:
                    result_image = stable_diffusion_text_to_image(
                        merged_prompt,
                        negative_prompt,
                        params.get("aspect_ratio", "1:1"),
                        params.get("seed", 0),
                        params.get("output_format", "jpeg")
                    )
                    print("[DEBUG] stable_diffusion_text_to_image finished.")
                    # Save result image to static/images/
                    result_path = save_image_to_static(result_image)
                    filename = os.path.basename(result_path)
                    image_url = f"/static/images/{filename}"
                    params["merged_prompt"] = merged_prompt
                    return ChatResponse(
                        response=analysis["response"],
                        action="text_to_image",
                        image_url=image_url,
                        parameters=params
                    )
                except Exception as e:
                    print(f"[DEBUG] ERROR: Image generation failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
            
            else:
                print(f"[DEBUG] Method called: Unknown or clarification action: {action}")
                return ChatResponse(
                    response="Sorry, I couldn't understand your request. Could you please rephrase or be more specific?",
                    action=action,
                    parameters=params
                )
            
            # Save result image to static/images/
            print("[DEBUG] Saving result image to static/images...")
            result_path = save_image_to_static(result_image)
            filename = os.path.basename(result_path)
            image_url = f"/static/images/{filename}"
            print(f"[DEBUG] Returning image_url: {image_url}")
            
            response_text = analysis["response"]
            return ChatResponse(
                response=response_text,
                action=action,
                image_url=image_url,
                parameters=params
            )
            
        except Exception as e:
            print(f"[DEBUG] ERROR: Exception during action execution: {str(e)}")
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise HTTPException(status_code=500, detail=f"House image processing failed: {str(e)}")
            
    except Exception as e:
        print(f"[DEBUG] ERROR: Exception in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

@app.get("/list-house-images")
def list_house_images():
    """List all house images in 'House Images/' as URLs for the frontend sidebar."""
    house_dir = os.path.join(os.getcwd(), "House Images")
    if not os.path.exists(house_dir):
        return {"images": []}
    image_files = [f for f in os.listdir(house_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    image_urls = [f"/House Images/{fname}" for fname in image_files]
    return {"images": image_urls}

# Serve House Images as static files
from fastapi.staticfiles import StaticFiles
app.mount("/House Images", StaticFiles(directory="House Images"), name="house-images")

# Serve static files and templates (you'll need to create these directories)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_app(request: Request):
        """Serve the main application frontend at root URL"""
        return templates.TemplateResponse("index.html", {"request": request})

@app.get("/style-transfer-page", response_class=HTMLResponse)
async def style_transfer_page(request: Request):
    return templates.TemplateResponse("style_transfer.html", {"request": request})

@app.get("/mask-page", response_class=HTMLResponse)
async def mask_page(request: Request):
    return templates.TemplateResponse("mask.html", {"request": request})

@app.get("/image-to-text", response_class=HTMLResponse)
def image_to_text_page(request: Request):
    return templates.TemplateResponse("text_to_image.html", {"request": request})

@app.post("/text-to-image")
async def text_to_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    aspect_ratio: str = Form("1:1"),
    seed: int = Form(0),
    output_format: str = Form("jpeg")
):
    try:
        print(f"[TEXT-TO-IMAGE] Received prompt: {prompt}")
        combined_prompt = f"{prompt}. {BASE_ARCHITECTURAL_PROMPT}"
        print(f"[TEXT-TO-IMAGE] Combined prompt: {combined_prompt}")
        result_image = stable_diffusion_text_to_image(
            combined_prompt,
            negative_prompt,
            aspect_ratio,
            seed,
            output_format
        )
        result_base64 = base64.b64encode(result_image).decode()
        return {
            "success": True,
            "image_data": f"data:image/{output_format};base64,{result_base64}",
            "prompt": combined_prompt
        }
    except Exception as e:
        print(f"[TEXT-TO-IMAGE][ERROR] {str(e)}")
        return {"success": False, "error": str(e)}

# Style Transfer Endpoint (moved from style_transfer_api.py)
@app.post("/style-transfer")
async def style_transfer(
    target_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    style_strength: float = Form(0.7),
    composition_fidelity: float = Form(0.95),
    change_strength: float = Form(0.7),
    prompt: str = Form(""),
    negative_prompt: str = Form("blurry, distorted, deformed, low quality, artifacts"),
    seed: int = Form(0),
    output_format: str = Form("jpeg")
):
    print("[STYLE-TRANSFER] Received request")
    print(f"[STYLE-TRANSFER] target_image.filename: {target_image.filename}, style_image.filename: {style_image.filename}")
    print(f"[STYLE-TRANSFER] style_strength: {style_strength}, composition_fidelity: {composition_fidelity}, change_strength: {change_strength}, prompt: {prompt}, negative_prompt: {negative_prompt}, seed: {seed}, output_format: {output_format}")
    try:
        target_bytes = await target_image.read()
        style_bytes = await style_image.read()
        print(f"[STYLE-TRANSFER] target_bytes size: {len(target_bytes)} bytes, style_bytes size: {len(style_bytes)} bytes")
        target_pil = Image.open(io.BytesIO(target_bytes)).convert("RGB")
        style_pil = Image.open(io.BytesIO(style_bytes)).convert("RGB")

        # Prepare files for API
        target_io = io.BytesIO()
        target_pil.save(target_io, format='JPEG')
        target_io.seek(0)
        style_io = io.BytesIO()
        style_pil.save(style_io, format='JPEG')
        style_io.seek(0)

        files = {
            "init_image": ("target.jpg", target_io, "image/jpeg"),
            "style_image": ("style.jpg", style_io, "image/jpeg")
        }
        params = {
            "change_strength": change_strength,
            "composition_fidelity": composition_fidelity,
            "output_format": output_format,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "style_strength": style_strength,
        }
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {STABILITY_API_KEY}"
        }
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/control/style-transfer",
            headers=headers,
            files=files,
            data=params
        )
        print(f"[STYLE-TRANSFER] API response status: {response.status_code}")
        if not response.ok:
            print(f"[STYLE-TRANSFER][ERROR] {response.text}")
            return JSONResponse(status_code=response.status_code, content={"error": response.text})
        # Return image and metadata
        result_bytes = io.BytesIO(response.content)
        finish_reason = response.headers.get("finish-reason", "completed")
        result_seed = response.headers.get("seed", "unknown")
        print(f"[STYLE-TRANSFER] finish_reason: {finish_reason}, result_seed: {result_seed}, result_bytes size: {result_bytes.getbuffer().nbytes} bytes")
        return StreamingResponse(
            result_bytes,
            media_type=f"image/{output_format}",
            headers={
                "finish-reason": finish_reason,
                "seed": str(result_seed)
            }
        )
    except Exception as e:
        print(f"[STYLE-TRANSFER][ERROR] {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/style-transfer")
def redirect_style_transfer():
    return RedirectResponse(url="/style-transfer-page")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
