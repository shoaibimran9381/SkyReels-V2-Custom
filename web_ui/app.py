"""
SkyReels V2 Web UI Backend Server
Handles video generation requests from the web interface
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc

# Serve static files (index.html, styles.css, script.js) directly from this folder
app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = '../result'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Cache for loaded models
model_cache = {}

def get_resolution_dims(resolution):
    """Convert resolution string to height and width"""
    if resolution == "540P":
        return 544, 960
    elif resolution == "720P":
        return 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

def generate_t2v(data):
    """Generate video from text prompt"""
    prompt = data.get('prompt')
    model_id = data.get('model_id')
    resolution = data.get('resolution', '540P')
    num_frames = int(data.get('num_frames', 97))
    inference_steps = int(data.get('inference_steps', 30))
    guidance_scale = float(data.get('guidance_scale', 6.0))
    shift = float(data.get('shift', 8.0))
    seed = data.get('seed')
    fps = int(data.get('fps', 24))
    prompt_enhancer = data.get('prompt_enhancer', False) == 'true'
    
    height, width = get_resolution_dims(resolution)

    import imageio
    import torch
    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, PromptEnhancer

    # Download model if needed
    model_id = download_model(model_id)
    
    # Set seed
    if seed:
        seed = int(seed)
    else:
        seed = int(random.randrange(4294967294))
    
    # Initialize pipeline
    pipe = Text2VideoPipeline(
        model_path=model_id,
        dit_path=model_id,
        use_usp=False,
        offload=False
    )

    # Enhance prompt if requested
    prompt_input = prompt
    if prompt_enhancer:
        try:
            enhancer = PromptEnhancer()
            prompt_input = enhancer(prompt)
            del enhancer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Prompt enhancement failed: {e}")
    
    # Generate video
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            height=height,
            width=width,
        )[0]
    
    # Save video
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'text2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"t2v_{prompt[:50].replace('/', '')}_{seed}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    
    return {
        'success': True,
        'output_path': output_path,
        'message': f'Video generated successfully: {video_file}'
    }

def generate_i2v(data, files):
    """Generate video from image"""
    prompt = data.get('prompt', '')
    model_id = data.get('model_id')
    resolution = data.get('resolution', '540P')
    num_frames = int(data.get('num_frames', 97))
    inference_steps = int(data.get('inference_steps', 30))
    guidance_scale = float(data.get('guidance_scale', 6.0))
    seed = data.get('seed')
    
    height, width = get_resolution_dims(resolution)
    
    # Handle image upload
    if 'image' not in files:
        return {'success': False, 'error': 'No image file provided'}
    
    image_file = files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(image_path)

    import imageio
    import torch
    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Image2VideoPipeline
    from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
    from diffusers.utils import load_image

    # Download model if needed
    model_id = download_model(model_id)
    
    # Set seed
    if seed:
        seed = int(seed)
    else:
        seed = int(random.randrange(4294967294))
    
    # Load image
    image = load_image(image_path).convert("RGB")
    image_width, image_height = image.size
    
    if image_height > image_width:
        height, width = width, height
    
    image = resizecrop(image, height, width)
    
    # Initialize pipeline
    pipe = Image2VideoPipeline(
        model_path=model_id,
        dit_path=model_id,
        use_usp=False,
        offload=False
    )

    # Generate video
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(
            image=image,
            prompt=prompt if prompt else "A video of this image",
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            height=height,
            width=width,
        )[0]
    
    # Save video
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'image2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"i2v_{prompt[:30] if prompt else 'generated'}_{seed}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    imageio.mimwrite(output_path, video_frames, fps=24, quality=8, output_params=["-loglevel", "error"])
    
    # Cleanup
    os.remove(image_path)
    
    return {
        'success': True,
        'output_path': output_path,
        'message': f'Video generated successfully: {video_file}'
    }

@app.route('/')
def index():
    """Serve the main web interface (index.html in this folder)."""
    return app.send_static_file('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handle video generation requests"""
    try:
        generation_type = request.form.get('type')
        
        if generation_type == 't2v':
            result = generate_t2v(request.form)
        elif generation_type == 'i2v':
            result = generate_i2v(request.form, request.files)
        elif generation_type == 'df':
            return jsonify({'success': False, 'error': 'Diffusion Forcing not yet implemented in web UI'})
        else:
            return jsonify({'success': False, 'error': 'Invalid generation type'})
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'SkyReels V2 Web UI is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Python executable: {sys.executable}")
    print(f"""
    ╔════════════════════════════════════════╗
    ║   SkyReels V2 Web UI Server            ║
    ║   Starting on http://localhost:{port}    ║
    ╚════════════════════════════════════════╝
    """)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
