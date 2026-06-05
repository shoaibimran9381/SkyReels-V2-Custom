"""
SkyReels V2 Web UI Backend - MOCK VERSION FOR MACOS TESTING
This is a mock server for testing the UI without GPU/CUDA requirements
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Serve static files directly from this folder
app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = '../result'

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def generate_t2v(data):
    """Mock: Generate video from text prompt"""
    prompt = data.get('prompt', 'No prompt')
    seed = data.get('seed', 'random')
    
    # Simulate processing time
    time.sleep(3)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'text2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"t2v_{prompt[:20]}_{seed}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    # Create a dummy file
    with open(output_path, 'w') as f:
        f.write(f"[MOCK VIDEO] Prompt: {prompt}\n")
    
    return {
        'success': True,
        'output_path': output_path,
        'message': f'✨ [MOCK] Video generated: {video_file}'
    }


def generate_i2v(data, files):
    """Mock: Generate video from image"""
    prompt = data.get('prompt', 'from image')
    
    if 'image' not in files:
        return {'success': False, 'error': 'No image file provided'}
    
    image_file = files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(image_path)
    
    # Simulate processing
    time.sleep(3)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'image2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"i2v_{prompt[:20]}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    with open(output_path, 'w') as f:
        f.write(f"[MOCK VIDEO] Image: {image_file.filename}\nPrompt: {prompt}\n")
    
    # Cleanup
    os.remove(image_path)
    
    return {
        'success': True,
        'output_path': output_path,
        'message': f'✨ [MOCK] Video generated from image: {video_file}'
    }


@app.route('/')
def index():
    """Serve the main web interface"""
    from flask import send_file
    return send_file('index.html', mimetype='text/html')


@app.route('/generate', methods=['POST'])
def generate():
    """Handle video generation requests (MOCK)"""
    try:
        gen_type = request.form.get('type')
        
        if gen_type == 't2v':
            result = generate_t2v(request.form)
        elif gen_type == 'i2v':
            result = generate_i2v(request.form, request.files)
        elif gen_type == 'df':
            return jsonify({'success': False, 'error': 'Diffusion Forcing not implemented'})
        else:
            return jsonify({'success': False, 'error': 'Invalid generation type'})
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'message': 'SkyReels V2 Web UI (MOCK MODE) is running on macOS',
        'mode': 'mock'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"🎬 SkyReels V2 Web UI - MOCK MODE (macOS)")
    print(f"{'='*60}")
    print(f"📱 Open browser: http://localhost:{port}")
    print(f"⚠️  Running in MOCK mode (no actual video generation)")
    print(f"{'='*60}\n")
    False, host='127.0.0.1', port=port, use_reloader=False
    app.run(debug=True, host='0.0.0.0', port=port)
