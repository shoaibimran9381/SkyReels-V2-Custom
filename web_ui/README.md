# SkyReels V2 Web UI

A modern web-based interface for the SkyReels V2 video generation model.

## Features

- 🎬 **Text-to-Video (T2V)** - Generate videos from text prompts
- 🖼️ **Image-to-Video (I2V)** - Generate videos from images
- 🎞️ **Video Extension** - Extend existing videos (Diffusion Forcing)
- ⚙️ **Advanced Controls** - Fine-tune generation parameters
- ✨ **Prompt Enhancement** - Auto-enhance prompts for better results
- 🎨 **Modern UI** - Beautiful, responsive dark theme

## Installation

1. **Clone and Navigate**
```bash
cd /path/to/SkyReels-V2-main/web_ui
```

2. **Install Web UI Dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Main Project Dependencies** (if not already installed)
```bash
pip install -r ../requirements.txt
```

## Usage

### Start the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### Using the Web Interface

1. **Open Browser**
   - Navigate to `http://localhost:5000`

2. **Select Generation Type**
   - Text-to-Video: Generate from text
   - Image-to-Video: Generate from image
   - Video Extension: Extend existing videos

3. **Configure Parameters**
   - Prompt/Image: Your input
   - Model: Choose resolution and capability
   - Quality Settings: Frames, steps, guidance scale, etc.

4. **Generate**
   - Click "Generate Video"
   - Wait for processing (this may take several minutes depending on your hardware)
   - Video saves to `../result/` directory

## Parameters Explained

### Text-to-Video
- **Prompt**: Detailed description of the video you want
- **Model**: 
  - 540P (faster, lower quality)
  - 720P (higher quality, slower)
- **Frames**: Total frames in video (97 recommended)
- **Inference Steps**: Quality vs speed (30 good, 50 best)
- **Guidance Scale**: How closely to follow prompt (6.0 default)
- **Shift**: Diffusion shift parameter (8.0 default)
- **Seed**: For reproducible results
- **Prompt Enhancement**: Auto-enhance your prompt

### Image-to-Video
- **Input Image**: Starting image
- **Prompt**: Optional motion description
- **Model**:
  - 1.3B (fast)
  - 14B (high quality)
- **Frame/Step Settings**: Similar to T2V

### Video Extension
- **Input Video**: Existing video to extend
- **Prompt**: How the video should continue
- **Overlap History**: Frames to overlap for consistency (17-37 recommended)

## Advanced Features

### Command Line Usage

Still prefer command line? Use the original scripts:

```bash
# Text-to-Video
python ../generate_video.py --prompt "Your prompt" --model_id "Skywork/SkyReels-V2-T2V-14B-540P"

# Image-to-Video
python ../generate_video.py --prompt "Your prompt" --image "path/to/image.jpg"

# Video Extension
python ../generate_video_df.py --prompt "Continuation" --video_path "path/to/video.mp4"
```

## Output

Generated videos are saved to:
```
../result/text2video/      (T2V outputs)
../result/image2video/     (I2V outputs)
../result/diffusion_forcing/ (DF outputs)
```

Filename format: `{type}_{prompt/seed}_{timestamp}.mp4`

## Troubleshooting

### CUDA Out of Memory
- Reduce number of frames
- Lower inference steps
- Use 540P instead of 720P
- Enable offload mode (in CLI)

### Slow Generation
- Use 1.3B models instead of 14B
- Reduce frames to 48-50
- Reduce inference steps to 20-25

### Model Download Issues
- Models auto-download from Hugging Face
- Ensure internet connection
- Check disk space (models are large ~10-30GB)

## Architecture

```
web_ui/
├── index.html          # Web interface
├── styles.css          # Modern dark theme styling
├── script.js           # Frontend interactions
├── app.py              # Flask backend server
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Performance Tips

1. **GPU**: NVIDIA GPU recommended (RTX 3080+)
2. **VRAM**: 16GB+ VRAM for best results
3. **Resolution**: 540P is faster, 720P is higher quality
4. **Batch Size**: Single video at a time (web UI)

## API Endpoints

- `GET /` - Serve web interface
- `POST /generate` - Generate video
- `GET /health` - Health check

## System Requirements

- Python 3.8+
- NVIDIA GPU (CUDA 11.8+)
- 32GB+ RAM
- 100GB+ disk space for models

## Next Steps

- Fine-tune parameters for your use case
- Experiment with different prompts
- Use Prompt Enhancement for better results
- Check out the original SkyReels papers for inspiration

## Support

- GitHub: https://github.com/SkyworkAI/SkyReels-V2
- Hugging Face: https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9
- Discord: https://discord.gg/PwM6NYtccQ

---

Happy video generating! 🎬✨
