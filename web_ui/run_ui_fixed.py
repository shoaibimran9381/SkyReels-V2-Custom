#!/usr/bin/env python3
"""
SkyReels V2 Web UI - MOCK VERSION FOR MACOS TESTING
"""

import os
import sys
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'result')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

WEB_UI_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== STATIC FILES ====================
def read_file(filename):
    """Read file from web_ui directory"""
    filepath = os.path.join(WEB_UI_DIR, filename)
    with open(filepath, 'r') as f:
        return f.read()


@app.route('/')
def serve_index():
    """Serve index.html"""
    content = read_file('index.html')
    return content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/styles.css')
def serve_css():
    """Serve CSS"""
    content = read_file('styles.css')
    return content, 200, {'Content-Type': 'text/css; charset=utf-8'}


@app.route('/script.js')
def serve_js():
    """Serve JavaScript"""
    content = read_file('script.js')
    return content, 200, {'Content-Type': 'application/javascript; charset=utf-8'}


# ==================== API ENDPOINTS ====================
def generate_t2v(data):
    """Mock: Generate video from text prompt"""
    prompt = data.get('prompt', 'No prompt')
    seed = data.get('seed', 'random')
    
    time.sleep(2)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'text2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"t2v_{prompt[:20].replace('/', '')}_{seed}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
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
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(image_path)
    
    time.sleep(2)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'image2video')
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_file = f"i2v_{prompt[:20]}_{current_time}.mp4"
    output_path = os.path.join(output_dir, video_file)
    
    with open(output_path, 'w') as f:
        f.write(f"[MOCK VIDEO]\nImage: {image_file.filename}\nPrompt: {prompt}\n")
    
    os.remove(image_path)
    
    return {
        'success': True,
        'output_path': output_path,
        'message': f'✨ [MOCK] Video generated from image: {video_file}'
    }


@app.route('/generate', methods=['POST'])
def generate():
    """Handle video generation requests"""
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
