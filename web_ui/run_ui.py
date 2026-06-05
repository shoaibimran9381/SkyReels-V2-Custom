#!/usr/bin/env python3
"""
SkyReels V2 Web UI - MOCK VERSION FOR MACOS TESTING
Simple web server that properly serves HTML, CSS, JS and handles API requests
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'result')

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Get the directory where this script is located
WEB_UI_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== STATIC FILES ====================
@app.route('/')
def serve_index():
    """Serve index.html"""
    return send_from_directory(WEB_UI_DIR, 'index.html', mimetype='text/html')


@app.route('/styles.css')
def serve_css():
    """Serve CSS file"""
    return send_from_directory(WEB_UI_DIR, 'styles.css', mimetype='text/css')


@app.route('/script.js')
def serve_js():
    """Serve JavaScript file"""
    return send_from_directory(WEB_UI_DIR, 'script.js', mimetype='application/javascript')


# ==================== API ENDPOINTS ====================
def generate_t2v(data):
    """Mock: Generate video from text prompt"""
    prompt = data.get('prompt', 'No prompt')
    seed = data.get('seed', 'random')
    
    # Simulate processing time
    time.sleep(2)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'text2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"t2v_{prompt[:20].replace('/', '')}_{seed}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    # Create a dummy file to simulate output
    with open(output_path, 'w') as f:
        f.write(f"[MOCK VIDEO]\nPrompt: {prompt}\nSeed: {seed}\n")
    
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
    from werkzeug.utils import secure_filename
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(image_path)
    
    # Simulate processing
    time.sleep(2)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'image2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"i2v_{prompt[:20]}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    with open(output_path, 'w') as f:
        f.write(f"[MOCK VIDEO]\nImage: {image_file.filename}\nPrompt: {prompt}\n")
    
    # Cleanup
    os.remove(image_path)
    
    return {
        'success': True,
        'output_path': output_path,
        'message': f'✨ [MOCK] Video generated from image: {video_file}'
    }


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
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'message': 'SkyReels V2 Web UI (MOCK MODE) is running',
        'mode': 'mock'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"   🎬 SkyReels V2 Web UI - MOCK MODE")
    print(f"{'='*70}")
    print(f"")
    print(f"   📱 Open browser: http://localhost:{port}")
    print(f"")
    print(f"   ⚠️  Running in MOCK mode (simulated video generation)")
    print(f"   🎯 For real inference: Need GPU (Linux/Windows)")
    print(f"")
    print(f"{'='*70}")
    print(f"{'='*70}\n")
    
    app.run(debug=False, host='127.0.0.1', port=port, use_reloader=False)
