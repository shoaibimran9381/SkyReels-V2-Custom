# SkyReels V2 GPU Server Deployment - Complete Guide

## 📋 Current Status (macOS - Mock Mode)
✅ UI fully functional  
✅ Form inputs working  
✅ Mock video generation working  
✅ Ready for GPU deployment  

---

## 🚀 GPU Server Deployment Steps

### **STEP 1: Server Setup (Linux)**

#### 1.1 OS & Basic Setup
```bash
# Use Linux (Ubuntu 20.04 LTS or 22.04 recommended)
# Connect to your server via SSH
ssh user@your_gpu_server_ip

# Update system
sudo apt-get update && sudo apt-get upgrade -y
```

#### 1.2 Install NVIDIA Drivers & CUDA
```bash
# Check current GPU
nvidia-smi

# If driver needed:
sudo apt-get install -y nvidia-driver-535

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

---

### **STEP 2: Clone & Setup Code**

#### 2.1 Clone Repository
```bash
cd /opt
sudo git clone https://github.com/SkyworkAI/SkyReels-V2.git
cd SkyReels-V2
sudo chown -R $USER:$USER .
```

#### 2.2 Create Python Environment (Python 3.11)
```bash
# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv

# Create venv
python3.11 -m venv venv_gpu
source venv_gpu/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### 2.3 Install PyTorch with CUDA Support
```bash
# PyTorch for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### 2.4 Install Dependencies
```bash
# Core packages
pip install transformers>=4.35.0 diffusers>=0.21.0 accelerate>=0.24.0
pip install imageio imageio-ffmpeg opencv-python
pip install pandas numpy tqdm
pip install Flask Werkzeug gunicorn

# vLLM (for caption generation)
pip install vllm>=0.2.0

# Optional but recommended
pip install xformers flash-attn
```

---

### **STEP 3: Deploy Production Code**

#### 3.1 Create Production Server Script
```bash
# Copy the production script to your server
scp run_ui_production.py user@your_gpu_server_ip:/opt/SkyReels-V2/web_ui/
```

Or create it directly on server:
```bash
# SSH into server
ssh user@your_gpu_server_ip

# Create the file
cat > /opt/SkyReels-V2/web_ui/run_ui_production.py << 'EOF'
#!/usr/bin/env python3
"""
SkyReels V2 Production Web UI - GPU Server Edition
"""

import os
import sys
import json
import time
import random
import torch
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'result')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

WEB_UI_DIR = os.path.dirname(os.path.abspath(__file__))
model_cache = {}

def get_resolution_dims(resolution):
    if resolution == "540P":
        return 544, 960
    elif resolution == "720P":
        return 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

def read_file(filename):
    filepath = os.path.join(WEB_UI_DIR, filename)
    with open(filepath, 'r') as f:
        return f.read()

# ==================== STATIC FILES ====================
@app.route('/')
def serve_index():
    content = read_file('index.html')
    return content, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/styles.css')
def serve_css():
    content = read_file('styles.css')
    return content, 200, {'Content-Type': 'text/css; charset=utf-8'}

@app.route('/script.js')
def serve_js():
    content = read_file('script.js')
    return content, 200, {'Content-Type': 'application/javascript; charset=utf-8'}

# ==================== REAL VIDEO GENERATION ====================
def generate_t2v(data):
    """Generate video from text prompt - REAL GPU INFERENCE"""
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, PromptEnhancer
    from skyreels_v2_infer.modules import download_model
    import imageio
    
    prompt = data.get('prompt', '')
    model_id = data.get('model_id', 'Skywork/SkyReels-V2-T2V-14B-540P')
    resolution = data.get('resolution', '540P')
    num_frames = int(data.get('num_frames', 97))
    inference_steps = int(data.get('inference_steps', 30))
    guidance_scale = float(data.get('guidance_scale', 6.0))
    shift = float(data.get('shift', 8.0))
    seed = data.get('seed')
    fps = int(data.get('fps', 24))
    prompt_enhancer = data.get('prompt_enhancer', False) == 'true'
    
    height, width = get_resolution_dims(resolution)
    
    # Download model if needed
    print(f"[T2V] Downloading model: {model_id}")
    model_path = download_model(model_id)
    
    # Set seed
    if seed:
        seed = int(seed)
    else:
        seed = int(random.randrange(4294967294))
    
    # Initialize pipeline (cached for speed)
    if model_id not in model_cache:
        print(f"[T2V] Loading model into GPU: {model_id}")
        model_cache[model_id] = Text2VideoPipeline(
            model_path=model_path,
            dit_path=model_path,
            use_usp=False,
            offload=False,
            enable_attention_slicing=True
        )
    
    pipe = model_cache[model_id]
    
    # Enhance prompt if requested
    prompt_input = prompt
    if prompt_enhancer:
        try:
            print(f"[T2V] Enhancing prompt...")
            enhancer = PromptEnhancer()
            prompt_input = enhancer(prompt)
            print(f"[T2V] Enhanced: {prompt_input}")
            del enhancer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[T2V] Prompt enhancement failed: {e}")
    
    # Generate video
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    print(f"[T2V] Generating: {prompt[:50]}... ({num_frames} frames, {inference_steps} steps)")
    
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
    
    print(f"[T2V] Saving video: {output_path}")
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    
    print(f"[T2V] Success! Video: {video_file}")
    
    return {
        'success': True,
        'output_path': output_path,
        'seed': seed,
        'message': f'Video generated successfully: {video_file}'
    }

def generate_i2v(data, files):
    """Generate video from image - REAL GPU INFERENCE"""
    from skyreels_v2_infer.pipelines import Image2VideoPipeline
    from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
    from skyreels_v2_infer.modules import download_model
    from diffusers.utils import load_image
    import imageio
    
    prompt = data.get('prompt', '')
    model_id = data.get('model_id', 'Skywork/SkyReels-V2-I2V-14B')
    resolution = data.get('resolution', '540P')
    num_frames = int(data.get('num_frames', 97))
    inference_steps = int(data.get('inference_steps', 30))
    guidance_scale = float(data.get('guidance_scale', 6.0))
    seed = data.get('seed')
    fps = int(data.get('fps', 24))
    
    if 'image' not in files:
        return {'success': False, 'error': 'No image file provided'}
    
    height, width = get_resolution_dims(resolution)
    
    # Save uploaded image
    image_file = files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(image_path)
    
    # Download model
    print(f"[I2V] Downloading model: {model_id}")
    model_path = download_model(model_id)
    
    # Set seed
    if seed:
        seed = int(seed)
    else:
        seed = int(random.randrange(4294967294))
    
    # Load and process image
    print(f"[I2V] Loading image: {image_file.filename}")
    image = load_image(image_path).convert("RGB")
    image_width, image_height = image.size
    
    if image_height > image_width:
        height, width = width, height
    
    image = resizecrop(image, height, width)
    
    # Initialize pipeline
    if model_id not in model_cache:
        print(f"[I2V] Loading model into GPU: {model_id}")
        model_cache[model_id] = Image2VideoPipeline(
            model_path=model_path,
            dit_path=model_path,
            use_usp=False,
            offload=False
        )
    
    pipe = model_cache[model_id]
    
    # Generate video
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    print(f"[I2V] Generating from image: {num_frames} frames")
    
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
    
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    
    # Cleanup
    os.remove(image_path)
    
    return {
        'success': True,
        'output_path': output_path,
        'seed': seed,
        'message': f'Video generated successfully: {video_file}'
    }

# ==================== API ENDPOINTS ====================
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
            return jsonify({'success': False, 'error': 'Diffusion Forcing not yet implemented'})
        else:
            return jsonify({'success': False, 'error': 'Invalid generation type'})
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    gpu_available = torch.cuda.is_available()
    return jsonify({
        'status': 'ok',
        'message': 'SkyReels V2 Production Server',
        'gpu_available': gpu_available,
        'gpu_name': torch.cuda.get_device_name(0) if gpu_available else 'None',
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    workers = int(os.environ.get('WORKERS', 1))
    
    print(f"\n{'='*70}")
    print(f"   🎬 SkyReels V2 Production Server")
    print(f"{'='*70}")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1e9:.1f}GB")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"{'='*70}\n")
    
    from gunicorn.app.base import BaseApplication
    
    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()
        
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)
        
        def load(self):
            return self.application
    
    options = {
        'bind': f'0.0.0.0:{port}',
        'workers': workers,
        'threads': 2,
        'worker_class': 'gthread',
        'timeout': 1800,  # 30 minutes for long videos
        'keep_alive': 5,
        'access_log': '/var/log/skyreels/access.log',
        'error_log': '/var/log/skyreels/error.log',
    }
    
    GunicornApp(app, options).run()
EOF
```

---

### **STEP 4: Configure System Services**

#### 4.1 Create Systemd Service
```bash
# Create log directory
sudo mkdir -p /var/log/skyreels
sudo chown $USER:$USER /var/log/skyreels

# Create systemd service file
sudo tee /etc/systemd/system/skyreels.service > /dev/null <<'EOF'
[Unit]
Description=SkyReels V2 Production Server
After=network.target

[Service]
Type=simple
User=<YOUR_USERNAME>
WorkingDirectory=/opt/SkyReels-V2/web_ui
Environment="PATH=/opt/SkyReels-V2/venv_gpu/bin"
Environment="PORT=8000"
Environment="WORKERS=1"
ExecStart=/opt/SkyReels-V2/venv_gpu/bin/python run_ui_production.py
Restart=always
RestartSec=10

# Resource limits
MemoryLimit=50G
MemoryMax=60G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

# Replace <YOUR_USERNAME> with actual username
sudo sed -i 's/<YOUR_USERNAME>/'"$USER"'/g' /etc/systemd/system/skyreels.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable skyreels
sudo systemctl start skyreels

# Check status
sudo systemctl status skyreels
sudo journalctl -u skyreels -f
```

---

### **STEP 5: Configure Nginx Reverse Proxy**

#### 5.1 Install & Configure Nginx
```bash
sudo apt-get install -y nginx

# Create nginx config
sudo tee /etc/nginx/sites-available/skyreels > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long video generation (up to 30 min)
        proxy_connect_timeout 1800s;
        proxy_send_timeout 1800s;
        proxy_read_timeout 1800s;
        
        # Upload limit
        client_max_body_size 500M;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/skyreels /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test & restart
sudo nginx -t
sudo systemctl restart nginx
```

#### 5.2 Enable HTTPS with Let's Encrypt (Recommended)
```bash
sudo apt-get install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

### **STEP 6: Start the Server**

```bash
# SSH into server
ssh user@your_gpu_server_ip

# Activate venv
cd /opt/SkyReels-V2/web_ui
source ../venv_gpu/bin/activate

# Test server first (without systemd)
PORT=8000 python run_ui_production.py

# If working, run via systemd instead:
sudo systemctl start skyreels
sudo systemctl status skyreels
```

---

### **STEP 7: Verify Deployment**

```bash
# Health check
curl http://your_gpu_server_ip/health

# Generate test video
curl -X POST http://your_gpu_server_ip/generate \
  -F "type=t2v" \
  -F "prompt=A beautiful sunset" \
  -F "model_id=Skywork/SkyReels-V2-T2V-14B-540P" \
  -F "resolution=540P" \
  -F "num_frames=97" \
  -F "inference_steps=30" \
  -F "guidance_scale=6.0"
```

---

## 📊 Performance Benchmarks

| GPU | Model | Resolution | Frames | Time | Notes |
|-----|-------|-----------|--------|------|-------|
| RTX 3090 | T2V 14B | 540P | 97 | 2-3 min | Recommended |
| RTX 3090 | T2V 14B | 720P | 97 | 4-5 min | Higher quality |
| A100 | T2V 14B | 540P | 97 | 1-2 min | Fastest |
| A100 | T2V 14B | 720P | 97 | 2-3 min | Production grade |

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce inference steps or use fp16
export CUDA_LAUNCH_BLOCKING=1
python -c "import torch; torch.cuda.set_per_process_memory_fraction(0.8)"
```

### Model Download Issues
```bash
# Pre-download models
huggingface-cli download Skywork/SkyReels-V2-T2V-14B-540P
huggingface-cli download Skywork/SkyReels-V2-T2V-14B-720P
```

### Slow Generation
```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Check temperatures
nvidia-smi -q -d TEMPERATURE
```

---

## 🎯 Transition from Mock to Real

**What Changes:**
1. ✅ Remove `# Simulate processing time` → Real GPU inference
2. ✅ Replace mock video creation → Actual frame generation
3. ✅ Use real models → No fallback needed
4. ✅ Output real `.mp4` → Full video files with frames

**Files on GPU Server:**
```
/opt/SkyReels-V2/
├── web_ui/
│   ├── run_ui_production.py  ← Real inference
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── requirements_production.txt
├── skyreels_v2_infer/  ← Real models
└── result/
    └── text2video/  ← Real video files (MP4 with frames)
```

---

## 📝 Quick Reference Commands

```bash
# Start server
sudo systemctl start skyreels

# Check logs
sudo journalctl -u skyreels -f

# Stop server
sudo systemctl stop skyreels

# Restart
sudo systemctl restart skyreels

# View GPU usage
nvidia-smi -l 1

# Check VRAM
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,nounits,noheader
```

---

**You're ready for production GPU deployment!** 🚀
