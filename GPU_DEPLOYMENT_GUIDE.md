# SkyReels V2 GPU Server Deployment Guide

## 🚀 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3090, A100, H100, etc.)
- **VRAM**: Minimum 24GB (40GB recommended for higher resolutions)
- **CPU**: 8+ cores
- **Storage**: 100GB+ (for model checkpoints)
- **RAM**: 64GB+ system RAM

### Software Requirements
```bash
# OS: Linux (Ubuntu 20.04 LTS or later recommended)
# Python: 3.10 or 3.11 (NOT 3.12+)
# CUDA: 11.8+ or 12.0+
# cuDNN: 8.x
```

---

## 📋 Step 1: Server Setup

### 1.1 Install CUDA & cuDNN
```bash
# Check NVIDIA driver
nvidia-smi

# If needed, install NVIDIA driver
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### 1.2 Clone Repository
```bash
cd /opt
sudo git clone https://github.com/SkyworkAI/SkyReels-V2.git
cd SkyReels-V2
sudo chown -R $USER:$USER .
```

### 1.3 Create Python Environment
```bash
# Use Python 3.11
python3.11 -m venv venv_gpu
source venv_gpu/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## 📦 Step 2: Install Dependencies

### 2.1 Core ML Stack
```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers and related
pip install transformers>=4.30 diffusers accelerate

# Video processing
pip install imageio imageio-ffmpeg

# Other dependencies
pip install pandas numpy tqdm opencv-python
```

### 2.2 vLLM (for production caption generation)
```bash
# vLLM with CUDA 12.1
pip install vllm[flashinfer]>=0.2.0

# Or with specific CUDA version
pip install vllm --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 Web Framework
```bash
pip install Flask Werkzeug gunicorn
```

### 2.4 Download Models
```bash
# Models are auto-downloaded on first use, but pre-download for faster startup:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Skywork/SkyReels-V2-T2V-14B-540P')"
```

---

## 🔧 Step 3: Modify Code for Production

### 3.1 Update `web_ui/run_ui_production.py`

Create a production version that uses real models:

```python
#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import torch
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'result')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

WEB_UI_DIR = os.path.dirname(os.path.abspath(__file__))

# Global model cache
model_cache = {}

def get_resolution_dims(resolution):
    """Convert resolution string to height and width"""
    if resolution == "540P":
        return 544, 960
    elif resolution == "720P":
        return 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

# ==================== STATIC FILES ====================
def read_file(filename):
    """Read file from web_ui directory"""
    filepath = os.path.join(WEB_UI_DIR, filename)
    with open(filepath, 'r') as f:
        return f.read()

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
    """Generate video from text prompt using real models"""
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
    model_path = download_model(model_id)
    
    # Set seed
    if seed:
        seed = int(seed)
    else:
        seed = int(random.randrange(4294967294))
    
    # Initialize pipeline (cached if possible)
    if model_id not in model_cache:
        print(f"Loading model: {model_id}")
        model_cache[model_id] = Text2VideoPipeline(
            model_path=model_path,
            dit_path=model_path,
            use_usp=False,
            offload=False
        )
    
    pipe = model_cache[model_id]
    
    # Enhance prompt if requested
    prompt_input = prompt
    if prompt_enhancer:
        try:
            enhancer = PromptEnhancer()
            prompt_input = enhancer(prompt)
            print(f"Enhanced prompt: {prompt_input}")
            del enhancer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Prompt enhancement failed: {e}")
    
    # Generate video
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    print(f"Generating video: {prompt[:50]}...")
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
    
    print(f"Saving video to: {output_path}")
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    
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
    gpu_available = torch.cuda.is_available()
    return jsonify({
        'status': 'ok',
        'message': 'SkyReels V2 Production Server',
        'gpu_available': gpu_available,
        'gpu_name': torch.cuda.get_device_name(0) if gpu_available else 'None',
        'cuda_version': torch.version.cuda
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
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"{'='*70}\n")
    
    # Use gunicorn for production
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
        'timeout': 900,
        'keep_alive': 5,
    }
    
    GunicornApp(app, options).run()
```

---

## 🚀 Step 4: Deploy & Run

### 4.1 Using Gunicorn
```bash
cd /opt/SkyReels-V2/web_ui

# Run with 4 workers (adjust based on GPU memory)
PORT=8000 WORKERS=1 python run_ui_production.py
```

### 4.2 Using systemd Service (Recommended)
```bash
# Create systemd service
sudo tee /etc/systemd/system/skyreels.service > /dev/null <<EOF
[Unit]
Description=SkyReels V2 Web UI
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/SkyReels-V2/web_ui
Environment="PATH=/opt/SkyReels-V2/venv_gpu/bin"
Environment="PORT=8000"
Environment="WORKERS=1"
ExecStart=/opt/SkyReels-V2/venv_gpu/bin/python run_ui_production.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable skyreels
sudo systemctl start skyreels

# Check status
sudo systemctl status skyreels
sudo journalctl -u skyreels -f
```

### 4.3 Using Docker (Best Practice)
```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/skyreels

COPY . .

RUN python3.11 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r web_ui/requirements_production.txt

ENV PATH="/opt/skyreels/venv/bin:$PATH"
ENV PORT=8000
ENV WORKERS=1

EXPOSE 8000

CMD ["python", "web_ui/run_ui_production.py"]
```

Build and run:
```bash
docker build -t skyreels-v2 .
docker run --gpus all -p 8000:8000 -v /data:/opt/skyreels/result skyreels-v2
```

---

## 🔒 Step 5: Production Configuration

### 5.1 Add Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name skyreels.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for long video generation
        proxy_connect_timeout 900s;
        proxy_send_timeout 900s;
        proxy_read_timeout 900s;
    }
}
```

### 5.2 Enable HTTPS with Let's Encrypt
```bash
sudo certbot --nginx -d skyreels.yourdomain.com
```

### 5.3 Resource Limits
```bash
# Edit /etc/systemd/system/skyreels.service
[Service]
MemoryLimit=50G
MemoryMax=60G
CPUQuota=80%
CPUAffinity=0-7
```

---

## 📊 Step 6: Optimization Tips

### 6.1 GPU Optimization
```python
# In run_ui_production.py - enable flash attention
from skyreels_v2_infer.pipelines import Text2VideoPipeline

pipe = Text2VideoPipeline(
    model_path=model_path,
    use_usp=False,  # Enable for slight quality improvement
    offload=False,  # Keep in VRAM for speed
    enable_flash_attention=True,  # Faster inference
    dtype=torch.float16  # Use half precision
)
```

### 6.2 Batch Processing
```python
# Queue multiple requests
import queue
request_queue = queue.Queue(maxsize=5)
```

### 6.3 Model Caching
- Keep models loaded in VRAM between requests
- Implement request queuing to avoid conflicts
- Use memory pooling for efficiency

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Generate video
curl -X POST http://localhost:8000/generate \
  -F "type=t2v" \
  -F "prompt=A beautiful sunset" \
  -F "model_id=Skywork/SkyReels-V2-T2V-14B-540P" \
  -F "resolution=540P" \
  -F "num_frames=97" \
  -F "inference_steps=30" \
  -F "guidance_scale=6.0"
```

---

## 📈 Performance Benchmarks

| Resolution | GPU | Time | VRAM |
|-----------|-----|------|------|
| 540P | RTX 3090 | 2-3 min | 22GB |
| 720P | RTX 3090 | 4-5 min | 24GB |
| 540P | A100 | 1-2 min | 30GB |
| 720P | A100 | 2-3 min | 40GB |

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size or use gradient checkpointing
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
```

### Slow Generation
```bash
# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Real-time monitoring
```

### Model Download Issues
```bash
# Pre-download models manually
huggingface-cli download Skywork/SkyReels-V2-T2V-14B-540P
```

---

## 📝 Create requirements_production.txt
```
Flask==3.0.0
Werkzeug==3.0.1
gunicorn==21.2.0
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
transformers==4.35.0
diffusers==0.21.0
accelerate==0.24.0
imageio==2.33.1
imageio-ffmpeg==0.4.9
opencv-python==4.8.1
pandas==2.1.1
numpy==1.24.3
tqdm==4.66.1
vllm>=0.2.0
peft>=0.4.0
```

---

Done! Your SkyReels V2 is ready for production GPU deployment! 🚀
