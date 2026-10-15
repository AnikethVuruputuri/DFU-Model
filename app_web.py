"""
DFU-SIAM v3 Local Web Application.
Zero external web dependencies (uses Python's standard http.server + modern responsive HTML5/JS).

Usage:
    python app_web.py
    (Automatically opens http://localhost:5000 in your web browser)
"""

import os
import sys
import io
import json
import base64
import glob
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from PIL import Image

import torch
from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_dfu_transforms
from xai.unified import XAIExplainer
from xai.base import PATHOLOGY_CLASSES, WAGNER_GRADES, overlay_heatmap_on_image

# Global model and explainer instance
GLOBAL_MODEL = None
GLOBAL_EXPLAINER = None
GLOBAL_CHECKPOINT = "checkpoints_a1_full_balanced/best_checkpoint.pth"
DEVICE = get_device()


def get_available_checkpoints():
    cands = [
        "checkpoints_a1_full_balanced/best_checkpoint.pth",
        "checkpoints_a1_full/best_checkpoint.pth",
        "checkpoints_swin_tiny/best_checkpoint.pth"
    ]
    found = glob.glob("checkpoints*/best_checkpoint.pth")
    for f in found:
        norm = os.path.normpath(f).replace('\\', '/')
        if norm not in cands:
            cands.append(norm)
    return [c for c in cands if os.path.exists(c)]


def load_model(checkpoint_path):
    global GLOBAL_MODEL, GLOBAL_EXPLAINER, GLOBAL_CHECKPOINT
    print(f"[*] Loading model checkpoint: {checkpoint_path} on {DEVICE}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    arch = cfg.get('arch', 'full')

    model = DFUSiamV3(
        arch=arch, pretrained=False,
        shared_dim=cfg.get('shared_dim', None),
        path_dim=cfg.get('path_dim', None),
        depth_dim=cfg.get('depth_dim', None)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    explainer = XAIExplainer(model, device=DEVICE)
    GLOBAL_MODEL = model
    GLOBAL_EXPLAINER = explainer
    GLOBAL_CHECKPOINT = checkpoint_path
    print(f"[+] Model loaded successfully ({arch})!")


def pil_to_base64(pil_img, format="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DFU-SIAM v3 — Clinical Diagnostics & Explainable AI</title>
<style>
  :root {
    --bg-main: #0f172a;
    --card-bg: #1e293b;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent-blue: #0284c7;
    --accent-teal: #0d9488;
    --accent-purple: #7c3aed;
    --border-color: #334155;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-main); color: var(--text-primary); min-height: 100vh; padding: 24px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border-color); }
  .header h1 { font-size: 1.5rem; color: #38bdf8; }
  .header .badge { background: #1e293b; padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; border: 1px solid var(--border-color); }
  
  .grid-layout { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
  @media(max-width: 900px) { .grid-layout { grid-template-columns: 1fr; } }
  
  .card { background: var(--card-bg); border-radius: 12px; padding: 20px; border: 1px solid var(--border-color); margin-bottom: 20px; }
  .card h2 { font-size: 1.1rem; margin-bottom: 14px; color: #e2e8f0; }

  .upload-box {
    border: 2px dashed #475569; border-radius: 10px; padding: 24px; text-align: center;
    cursor: pointer; transition: border-color 0.2s; background: #0f172a;
  }
  .upload-box:hover { border-color: #38bdf8; }
  .preview-img { max-width: 100%; max-height: 220px; border-radius: 8px; margin-top: 12px; display: none; object-fit: contain; }

  select, button {
    width: 100%; padding: 12px; border-radius: 8px; margin-top: 8px;
    background: #0f172a; border: 1px solid var(--border-color); color: var(--text-primary); font-size: 0.95rem;
  }
  button {
    background: #0284c7; border: none; font-weight: bold; cursor: pointer; transition: background 0.2s; margin-top: 16px;
  }
  button:hover { background: #0369a1; }
  button:disabled { background: #475569; cursor: not-allowed; }

  .results-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
  .metric-card { background: #0f172a; border-radius: 8px; padding: 16px; border: 1px solid var(--border-color); text-align: center; }
  .metric-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: bold; }
  .metric-value { font-size: 1.4rem; font-weight: bold; margin: 8px 0; color: #38bdf8; }
  .metric-sub { font-size: 0.85rem; color: #cbd5e1; }

  .gallery-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
  .gallery-item { background: #0f172a; border-radius: 8px; padding: 12px; border: 1px solid var(--border-color); text-align: center; }
  .gallery-item h4 { font-size: 0.9rem; margin-bottom: 8px; color: #cbd5e1; }
  .gallery-item img { width: 100%; height: auto; border-radius: 6px; border: 1px solid #334155; }

  .spinner { display: none; text-align: center; padding: 20px; font-weight: bold; color: #38bdf8; }
</style>
</head>
<body>

<div class="header">
  <h1>DFU-SIAM v3: Clinical Diagnosis & Explainable AI (XAI)</h1>
  <div class="badge" id="deviceBadge">CUDA / GPU Ready</div>
</div>

<div class="grid-layout">
  <!-- Controls -->
  <div>
    <div class="card">
      <h2>1. Select Checkpoint</h2>
      <select id="ckptSelect"></select>
    </div>

    <div class="card">
      <h2>2. Upload Ulcer Image</h2>
      <div class="upload-box" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <p id="uploadPrompt">Drag & drop ulcer photo here, or <strong>Click to Browse</strong></p>
        <input type="file" id="fileInput" accept="image/*" style="display:none">
        <img id="previewImg" class="preview-img" alt="Preview">
      </div>
      <button id="btnAnalyze" onclick="runInference()" disabled>Run Diagnosis & Generate XAI</button>
      <div id="spinner" class="spinner">Processing image and computing gradient attributions...</div>
    </div>
  </div>

  <!-- Results -->
  <div>
    <div class="card">
      <h2>Clinical Diagnosis Summary</h2>
      <div class="results-grid">
        <div class="metric-card">
          <div class="metric-title">Pathology Class</div>
          <div class="metric-value" id="valPath">—</div>
          <div class="metric-sub" id="subPath">Confidence: —</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Wagner Depth Grade</div>
          <div class="metric-value" id="valWag" style="color:#0d9488">—</div>
          <div class="metric-sub" id="subWag">Confidence: —</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">University of Texas Matrix</div>
          <div class="metric-value" id="valUT" style="color:#a855f7">—</div>
          <div class="metric-sub" id="subUT">Joint Prob: —</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Explainable AI (XAI) Attributions</h2>
      <div class="gallery-grid" id="galleryGrid">
        <p style="color:var(--text-muted); grid-column: 1/-1;">Upload an image and run analysis to view visual heatmaps.</p>
      </div>
    </div>
  </div>
</div>

<script>
let uploadedBase64 = null;

// Initialize Checkpoint List
fetch('/api/checkpoints').then(r => r.json()).then(ckpts => {
  const sel = document.getElementById('ckptSelect');
  ckpts.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.innerText = c;
    sel.appendChild(opt);
  });
});

// File upload handling
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const previewImg = document.getElementById('previewImg');
const btnAnalyze = document.getElementById('btnAnalyze');

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if(file) processFile(file);
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#38bdf8'; });
dropZone.addEventListener('dragleave', e => { e.preventDefault(); dropZone.style.borderColor = '#475569'; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '#475569';
  if(e.dataTransfer.files.length > 0) processFile(e.dataTransfer.files[0]);
});

function processFile(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    uploadedBase64 = ev.target.result;
    previewImg.src = uploadedBase64;
    previewImg.style.display = 'block';
    document.getElementById('uploadPrompt').style.display = 'none';
    btnAnalyze.disabled = false;
  };
  reader.readAsDataURL(file);
}

function runInference() {
  if(!uploadedBase64) return;
  const btn = document.getElementById('btnAnalyze');
  const spinner = document.getElementById('spinner');
  btn.disabled = true;
  spinner.style.display = 'block';

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: uploadedBase64,
      checkpoint: document.getElementById('ckptSelect').value
    })
  })
  .then(r => r.json())
  .then(data => {
    spinner.style.display = 'none';
    btn.disabled = false;
    if(data.error) {
      alert("Error: " + data.error);
      return;
    }

    // Render Metrics
    document.getElementById('valPath').innerText = data.pathology.toUpperCase();
    document.getElementById('subPath').innerText = `Confidence: ${(data.pathology_conf * 100).toFixed(1)}%`;

    document.getElementById('valWag').innerText = data.wagner.toUpperCase();
    document.getElementById('subWag').innerText = `Confidence: ${(data.wagner_conf * 100).toFixed(1)}%`;

    document.getElementById('valUT').innerText = data.ut_stage;
    document.getElementById('subUT').innerText = `Joint Prob: ${(data.ut_prob * 100).toFixed(1)}%`;

    // Render Gallery
    const gallery = document.getElementById('galleryGrid');
    gallery.innerHTML = '';

    const items = [
      { title: "Original Ulcer Image", src: data.images.original },
      { title: "Combined Multi-Method Summary", src: data.images.summary },
      { title: "CNN Grad-CAM (EfficientNetV2)", src: data.images.gradcam_cnn },
      { title: "Swin Transformer Window Attention", src: data.images.transformer },
      { title: "Dual-Branch Fused Explanation", src: data.images.gradcam_fused },
      { title: "Integrated Gradients Path Attribution", src: data.images.integrated_gradients },
      { title: "KernelSHAP Superpixel Importance", src: data.images.shap }
    ];

    items.forEach(it => {
      if(it.src) {
        const div = document.createElement('div');
        div.className = 'gallery-item';
        div.innerHTML = `<h4>${it.title}</h4><img src="${it.src}" alt="${it.title}">`;
        gallery.appendChild(div);
      }
    });
  })
  .catch(err => {
    spinner.style.display = 'none';
    btn.disabled = false;
    alert("Inference request failed: " + err);
  });
}
</script>
</body>
</html>
"""


class DFURequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == "/api/checkpoints":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            ckpts = get_available_checkpoints()
            self.wfile.write(json.dumps(ckpts).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode('utf-8'))
                img_b64 = data.get('image_base64', '')
                ckpt_req = data.get('checkpoint', GLOBAL_CHECKPOINT)

                if ckpt_req != GLOBAL_CHECKPOINT or GLOBAL_MODEL is None:
                    load_model(ckpt_req)

                # Decode base64 image
                if "," in img_b64:
                    img_b64 = img_b64.split(",")[1]
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Run inference & explainability
                transform = get_dfu_transforms(is_train=False, img_size=(224, 224))
                results, meta = GLOBAL_EXPLAINER.explain_image(
                    pil_img=pil_img,
                    transform=transform,
                    target_type='pathology',
                    target_id='predicted',
                    method='all'
                )

                # Overlays to base64
                cnn_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['cnn_heatmap'])
                fused_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['fused_heatmap'])
                trans_ov, _ = overlay_heatmap_on_image(pil_img, results['transformer']['transformer_heatmap'])
                ig_ov, _ = overlay_heatmap_on_image(pil_img, results['integrated_gradients']['ig_heatmap'])
                shap_ov, _ = overlay_heatmap_on_image(pil_img, results['shap']['shap_heatmap'])

                # Summary figure
                sum_buf = io.BytesIO()
                GLOBAL_EXPLAINER._create_summary_figure(pil_img, results, meta, sum_buf)
                sum_buf.seek(0)
                sum_b64 = "data:image/png;base64," + base64.b64encode(sum_buf.getvalue()).decode('utf-8')

                p_idx = meta['pred_pathology_id']
                w_idx = meta['pred_wagner_grade']
                p_conf = max(meta['pathology_probs'])
                w_conf = max(meta['wagner_probs'])

                stage_letters = ['A', 'B', 'C', 'D']
                ut_stage = f"Stage {stage_letters[p_idx]} (Depth {min(w_idx, 3)})"
                ut_prob = p_conf * w_conf

                response_payload = {
                    'pathology': meta['pred_pathology_name'],
                    'pathology_conf': float(p_conf),
                    'wagner': meta['pred_wagner_name'],
                    'wagner_conf': float(w_conf),
                    'ut_stage': ut_stage,
                    'ut_prob': float(ut_prob),
                    'images': {
                        'original': pil_to_base64(pil_img),
                        'summary': sum_b64,
                        'gradcam_cnn': pil_to_base64(Image.fromarray(cnn_ov)),
                        'gradcam_fused': pil_to_base64(Image.fromarray(fused_ov)),
                        'transformer': pil_to_base64(Image.fromarray(trans_ov)),
                        'integrated_gradients': pil_to_base64(Image.fromarray(ig_ov)),
                        'shap': pil_to_base64(Image.fromarray(shap_ov))
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=5000):
    ckpts = get_available_checkpoints()
    if ckpts:
        load_model(ckpts[0])

    server_address = ('', port)
    httpd = HTTPServer(server_address, DFURequestHandler)
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f" DFU-SIAM v3 WEB APPLICATION RUNNING")
    print(f" URL: {url}")
    print(f" Press Ctrl+C to terminate the server")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port=port)
