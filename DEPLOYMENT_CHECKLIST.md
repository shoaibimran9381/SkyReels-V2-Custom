# GPU Deployment Quick Checklist

## ✅ Pre-Deployment (macOS - Already Done)

- [x] UI fully functional on localhost
- [x] Mock video generation working
- [x] Form inputs validated
- [x] File storage working
- [x] Code ready for GPU deployment

---

## 🚀 GPU Server Deployment Checklist

### Phase 1: Server Setup (1-2 hours)
- [ ] Rent GPU server (AWS, GCP, Azure, or on-premise)
  - GPU: RTX 3090 / A100 / RTX 4090
  - RAM: 64GB+
  - Storage: 100GB+
  - OS: Ubuntu 20.04 or 22.04

- [ ] SSH into server
  ```bash
  ssh user@your_gpu_server_ip
  ```

- [ ] Install NVIDIA drivers
  ```bash
  nvidia-smi  # Test installation
  ```

- [ ] Install CUDA 12.1
  ```bash
  nvcc --version  # Verify
  ```

### Phase 2: Environment Setup (30-45 minutes)
- [ ] Clone repository
  ```bash
  cd /opt && sudo git clone https://github.com/SkyworkAI/SkyReels-V2.git
  ```

- [ ] Create Python 3.11 venv
  ```bash
  python3.11 -m venv venv_gpu
  source venv_gpu/bin/activate
  ```

- [ ] Install PyTorch + CUDA
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- [ ] Install dependencies
  ```bash
  pip install -r web_ui/requirements_production.txt
  ```

### Phase 3: Deploy Code (10 minutes)
- [ ] Copy `run_ui_production.py` to GPU server
  ```bash
  scp run_ui_production.py user@your_ip:/opt/SkyReels-V2/web_ui/
  ```

- [ ] Test server
  ```bash
  PORT=8000 python run_ui_production.py
  ```

### Phase 4: Configure Services (20 minutes)
- [ ] Create systemd service
  ```bash
  sudo tee /etc/systemd/system/skyreels.service
  ```

- [ ] Start service
  ```bash
  sudo systemctl start skyreels
  sudo systemctl status skyreels
  ```

### Phase 5: Configure Nginx & HTTPS (15 minutes)
- [ ] Install Nginx
  ```bash
  sudo apt-get install nginx
  ```

- [ ] Configure reverse proxy
  ```bash
  sudo tee /etc/nginx/sites-available/skyreels
  sudo systemctl restart nginx
  ```

- [ ] Setup HTTPS (optional but recommended)
  ```bash
  sudo certbot --nginx -d yourdomain.com
  ```

### Phase 6: Testing (10 minutes)
- [ ] Health check
  ```bash
  curl http://your_ip/health
  ```

- [ ] Generate test video
  ```bash
  curl -X POST http://your_ip/generate \
    -F "type=t2v" \
    -F "prompt=A beautiful sunset" \
    -F "model_id=Skywork/SkyReels-V2-T2V-14B-540P"
  ```

- [ ] Check GPU usage
  ```bash
  nvidia-smi
  watch -n 1 nvidia-smi
  ```

---

## 🔑 Key Differences: Mock → GPU

| Component | Mock (macOS) | GPU Server |
|-----------|------------|-----------|
| **Inference** | Simulated | Real GPU computation |
| **Video Output** | Text files | Actual MP4 with frames |
| **Generation Time** | Instant (2 sec) | 2-5 minutes (540P) |
| **Model Loading** | N/A | 30GB+ from HuggingFace |
| **VRAM Required** | N/A | 24GB+ (RTX 3090) |
| **Output Size** | ~200 bytes | 50-500 MB per video |

---

## 📊 Estimated Costs

### AWS Pricing
```
g4dn.12xlarge (GPU server):
- $4.48/hour × 730 hours = ~$3,270/month
- OR p3.2xlarge: $3.06/hour = ~$2,230/month

Bandwidth:
- Outbound: $0.02 per GB
- 10GB/month videos = $0.20
```

### GCP Pricing
```
1x NVIDIA A100 GPU:
- $3.67/hour × 730 hours = ~$2,680/month

Or rent pre-configured instance for cheaper rates
```

### Azure Pricing
```
NC6s_v3 (RTX 2080):
- $1.26/hour = ~$920/month (economical)

ND40rs_v2 (A100):
- $5.04/hour = ~$3,680/month
```

---

## 🛠️ Server Recommendations

### Budget Setup (~$1,000/month)
- AWS EC2 g4dn.xlarge or p3.2xlarge
- 1x RTX 2080 or V100
- Generate ~100 videos/month

### Production Setup (~$3,000-5,000/month)
- AWS g4dn.12xlarge or p3.8xlarge
- RTX 3090 or A100
- Generate ~500-1000 videos/month
- Auto-scaling, multiple GPUs

### Enterprise Setup (~$10,000+/month)
- Multi-GPU setup (4x A100)
- Kubernetes orchestration
- Load balancing
- 24/7 monitoring

---

## 📞 Support Services

If you get stuck, these services can help:

1. **Model Hub Setup**: HuggingFace model download issues
   - https://huggingface.co/docs

2. **CUDA/GPU Issues**: NVIDIA documentation
   - https://docs.nvidia.com/cuda/

3. **Infrastructure**: AWS/GCP support
   - AWS: https://aws.amazon.com/support/
   - GCP: https://cloud.google.com/support

---

## ✨ Final Checklist Before Going Live

- [ ] GPU server is online and accessible
- [ ] All dependencies installed and tested
- [ ] Health endpoint returns `gpu_available: true`
- [ ] Test generation completes successfully
- [ ] Videos are saved with correct metadata
- [ ] Nginx reverse proxy working
- [ ] HTTPS certificate installed (production)
- [ ] Monitoring set up (GPU temperature, memory)
- [ ] Logs configured and accessible
- [ ] Firewall rules configured
- [ ] Backup strategy in place
- [ ] Documentation updated

---

## 🎯 Next Steps

1. **Choose GPU Server Provider** → AWS/GCP/Azure/On-premise
2. **Launch Server Instance** → Select GPU, OS, storage
3. **Follow PRODUCTION_DEPLOYMENT.md** → Step-by-step setup
4. **Test with Small Videos** → 100-200 frames first
5. **Scale to Production** → Increase frame counts, optimize
6. **Monitor Performance** → Track GPU usage, generation times
7. **Iterate & Improve** → Gather user feedback, optimize

---

**Total Deployment Time: 3-4 hours**  
**Total Setup Cost: ~$50-200 (depending on provider)**  
**Monthly Operating Cost: $900-5,000 (depending on usage)**

You've got this! 🚀
