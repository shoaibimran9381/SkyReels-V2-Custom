// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Remove active class from all buttons
        document.querySelectorAll('.tab-btn').forEach(b => {
            b.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(tabName).classList.add('active');
        btn.classList.add('active');
    });
});

// Duration Calculator
function updateDuration() {
    const frames = parseInt(document.getElementById('t2v-frames').value) || 97;
    const fps = parseInt(document.getElementById('t2v-fps').value) || 24;
    const duration = (frames / fps).toFixed(1);
    const durationElement = document.getElementById('t2v-duration');
    if (durationElement) {
        durationElement.textContent = `(${duration} sec)`;
    }
}

document.getElementById('t2v-frames').addEventListener('input', updateDuration);
document.getElementById('t2v-fps').addEventListener('input', updateDuration);

// Initialize duration on page load
updateDuration();

// Image Preview
document.getElementById('i2v-image').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const preview = document.getElementById('i2v-preview');
            preview.innerHTML = `<img src="${event.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
});

// T2V Form Submission
document.getElementById('t2vForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData();
    
    formData.append('type', 't2v');
    formData.append('prompt', document.getElementById('t2v-prompt').value);
    formData.append('model_id', document.getElementById('t2v-model').value);
    formData.append('resolution', document.getElementById('t2v-resolution').value);
    formData.append('num_frames', document.getElementById('t2v-frames').value);
    formData.append('inference_steps', document.getElementById('t2v-steps').value);
    formData.append('guidance_scale', document.getElementById('t2v-guidance').value);
    formData.append('shift', document.getElementById('t2v-shift').value);
    formData.append('seed', document.getElementById('t2v-seed').value || null);
    formData.append('fps', document.getElementById('t2v-fps').value);
    formData.append('prompt_enhancer', document.getElementById('t2v-enhancer').checked);
    
    submitForm(formData);
});

// I2V Form Submission
document.getElementById('i2vForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData();
    
    formData.append('type', 'i2v');
    formData.append('image', document.getElementById('i2v-image').files[0]);
    formData.append('prompt', document.getElementById('i2v-prompt').value);
    formData.append('model_id', document.getElementById('i2v-model').value);
    formData.append('resolution', document.getElementById('i2v-resolution').value);
    formData.append('num_frames', document.getElementById('i2v-frames').value);
    formData.append('inference_steps', document.getElementById('i2v-steps').value);
    formData.append('guidance_scale', document.getElementById('i2v-guidance').value);
    formData.append('seed', document.getElementById('i2v-seed').value || null);
    
    submitForm(formData);
});

// DF Form Submission
document.getElementById('dfForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData();
    
    formData.append('type', 'df');
    formData.append('video', document.getElementById('df-video').files[0]);
    formData.append('prompt', document.getElementById('df-prompt').value);
    formData.append('model_id', document.getElementById('df-model').value);
    formData.append('resolution', document.getElementById('df-resolution').value);
    formData.append('num_frames', document.getElementById('df-frames').value);
    formData.append('overlap_history', document.getElementById('df-overlap').value);
    formData.append('inference_steps', document.getElementById('df-steps').value);
    formData.append('guidance_scale', document.getElementById('df-guidance').value);
    
    submitForm(formData);
});

let progressInterval = null;
let progressPercent = 0;

// Submit Form to Backend
function submitForm(formData) {
    const statusMsg = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressFill = document.getElementById('progress-fill');
    
    clearProgress();
    progressPercent = 0;
    progressFill.style.width = '0%';
    progressBar.style.display = 'block';
    statusMsg.className = 'status-message info show';
    statusMsg.innerHTML = '⏳ Preparing your video generation...<br>📥 Downloading model: 0%';
    
    // Disable all forms
    document.querySelectorAll('form').forEach(form => {
        form.style.opacity = '0.6';
        form.style.pointerEvents = 'none';
    });
    
    progressInterval = window.setInterval(() => {
        if (progressPercent < 95) {
            progressPercent += Math.floor(Math.random() * 6) + 2;
            if (progressPercent > 95) progressPercent = 95;
            progressFill.style.width = `${progressPercent}%`;
            statusMsg.innerHTML = `⏳ Preparing your video generation...<br>📥 Downloading model: ${progressPercent}%`;
        }
    }, 700);
    
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            progressPercent = 100;
            progressFill.style.width = '100%';
            statusMsg.className = 'status-message success show';
            statusMsg.innerHTML = `✅ Video generated successfully!<br>📁 Saved to: <code>${data.output_path}</code>`;
        } else {
            statusMsg.className = 'status-message error show';
            statusMsg.textContent = `❌ Error: ${data.error}`;
        }
    })
    .catch(error => {
        statusMsg.className = 'status-message error show';
        statusMsg.textContent = `❌ Error: ${error.message}`;
    })
    .finally(() => {
        clearProgress();
        document.querySelectorAll('form').forEach(form => {
            form.style.opacity = '1';
            form.style.pointerEvents = 'auto';
        });
        if (progressPercent < 100) {
            progressFill.style.width = '100%';
        }
        window.setTimeout(() => {
            progressBar.style.display = 'none';
        }, 1200);
    });
}

function clearProgress() {
    if (progressInterval) {
        window.clearInterval(progressInterval);
        progressInterval = null;
    }
}

console.log('SkyReels V2 Web UI loaded successfully! 🎬');
