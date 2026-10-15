"""
DFU-SIAM v3 Desktop GUI Application.
Features:
- Image file selector / drag-and-drop preview
- Multi-task clinical diagnosis (DFUC Pathology, Wagner Grade, UT Wound Stage)
- Full Explainable AI (XAI) gallery:
  * CNN Grad-CAM
  * Swin Transformer Window Attention Attribution
  * Dual-Branch Fused Attribution
  * Integrated Gradients
  * Superpixel KernelSHAP
  * Side-by-Side Comparison Panel
- Responsive non-blocking UI using background threading
"""

import os
import sys
import glob
import threading
from pathlib import Path
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import torch.nn.functional as F

from dfu_siam_model import DFUSiamV3, get_device
from dataset import get_dfu_transforms
from xai.unified import XAIExplainer
from xai.base import PATHOLOGY_CLASSES, WAGNER_GRADES, overlay_heatmap_on_image


class DFUSiamApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DFU-SIAM v3 — Clinical Diagnostic & Explainable AI Assistant")
        self.geometry("1280x850")
        self.minsize(1050, 700)

        # Style configuration
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self.style.configure('TFrame', background='#f4f6f9')
        self.style.configure('TLabel', background='#f4f6f9', font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#1e293b')
        self.style.configure('Card.TFrame', background='#ffffff', relief='flat')
        self.style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'), background='#0284c7', foreground='#ffffff')
        self.style.map('Primary.TButton', background=[('active', '#0369a1')])

        self.configure(bg='#f4f6f9')

        # Model and explainer state
        self.model = None
        self.explainer = None
        self.device = get_device()
        self.selected_img_path = None
        self.current_pil_img = None
        self.cached_results = None
        self.cached_meta = None
        self.photo_references = {}

        self._build_ui()
        self._find_checkpoints()

    def _build_ui(self):
        # Top banner
        banner = tk.Frame(self, bg='#0f172a', height=65)
        banner.pack(fill=tk.X, side=tk.TOP)
        banner.pack_propagate(False)

        title_lbl = tk.Label(
            banner, text="DFU-SIAM v3: Medical Imaging Diagnostics & XAI",
            font=('Segoe UI', 15, 'bold'), fg='#f8fafc', bg='#0f172a'
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        dev_str = f"Device: {self.device}".upper()
        dev_lbl = tk.Label(
            banner, text=dev_str, font=('Segoe UI', 9, 'bold'),
            fg='#38bdf8', bg='#0f172a'
        )
        dev_lbl.pack(side=tk.RIGHT, padx=20, pady=15)

        # Main horizontal split
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left Control Panel
        left_frame = ttk.Frame(main_paned, width=380)
        main_paned.add(left_frame, weight=0)

        # Right Display Panel
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self._build_left_controls(left_frame)
        self._build_right_results(right_frame)

    def _build_left_controls(self, parent):
        # 1. Model Checkpoint Card
        ckpt_card = ttk.LabelFrame(parent, text=" 1. Model Configuration ", padding=10)
        ckpt_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ckpt_card, text="Checkpoint:").pack(anchor=tk.W)
        self.ckpt_var = tk.StringVar()
        self.ckpt_combo = ttk.Combobox(ckpt_card, textvariable=self.ckpt_var, state='readonly', font=('Segoe UI', 9))
        self.ckpt_combo.pack(fill=tk.X, pady=(2, 6))
        self.ckpt_combo.bind('<<ComboboxSelected>>', lambda e: self._on_checkpoint_change())

        self.btn_load_ckpt = ttk.Button(ckpt_card, text="Browse Custom Checkpoint...", command=self._browse_custom_checkpoint)
        self.btn_load_ckpt.pack(fill=tk.X)

        self.ckpt_status_lbl = ttk.Label(ckpt_card, text="Model: Not Loaded", foreground='#64748b', font=('Segoe UI', 9))
        self.ckpt_status_lbl.pack(anchor=tk.W, pady=(4, 0))

        # 2. Image Selection Card
        img_card = ttk.LabelFrame(parent, text=" 2. Test Image Input ", padding=10)
        img_card.pack(fill=tk.X, pady=(0, 10))

        self.btn_select_img = ttk.Button(img_card, text="📁 Choose Ulcer Image...", command=self._browse_image)
        self.btn_select_img.pack(fill=tk.X, pady=(0, 6))

        # Thumbnail canvas
        self.thumb_canvas = tk.Canvas(img_card, bg='#e2e8f0', width=220, height=200, relief='solid', borderwidth=1)
        self.thumb_canvas.pack(pady=4)
        self.thumb_placeholder = self.thumb_canvas.create_text(
            110, 100, text="No Image Selected\n(Click 'Choose Ulcer Image')",
            fill='#64748b', justify=tk.CENTER, font=('Segoe UI', 9)
        )

        self.img_info_lbl = ttk.Label(img_card, text="Resolution: -", foreground='#64748b', font=('Segoe UI', 9))
        self.img_info_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 3. Explainability Options
        xai_card = ttk.LabelFrame(parent, text=" 3. Diagnostics & XAI Settings ", padding=10)
        xai_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(xai_card, text="Taxonomy Target:").pack(anchor=tk.W)
        self.target_var = tk.StringVar(value="pathology")
        target_cb = ttk.Combobox(xai_card, textvariable=self.target_var, state='readonly',
                                 values=["pathology", "wagner"], font=('Segoe UI', 9))
        target_cb.pack(fill=tk.X, pady=(2, 6))

        ttk.Label(xai_card, text="Explainability Method:").pack(anchor=tk.W)
        self.method_var = tk.StringVar(value="all")
        method_cb = ttk.Combobox(xai_card, textvariable=self.method_var, state='readonly',
                                 values=["all", "gradcam", "transformer", "integrated_gradients", "shap"],
                                 font=('Segoe UI', 9))
        method_cb.pack(fill=tk.X, pady=(2, 6))

        # 4. Action Button & Progress
        action_card = ttk.Frame(parent)
        action_card.pack(fill=tk.X, pady=8)

        self.btn_run = ttk.Button(action_card, text="⚡ Run Diagnosis & XAI", style='Primary.TButton', command=self._start_analysis)
        self.btn_run.pack(fill=tk.X, ipady=6)

        self.progress_bar = ttk.Progressbar(action_card, mode='indeterminate')
        self.status_lbl = ttk.Label(action_card, text="Ready", foreground='#475569', font=('Segoe UI', 9, 'italic'))
        self.status_lbl.pack(anchor=tk.W, pady=(4, 0))

    def _build_right_results(self, parent):
        # Diagnostic Summary Header Cards
        summary_frame = tk.Frame(parent, bg='#ffffff', relief='solid', borderwidth=1)
        summary_frame.pack(fill=tk.X, pady=(0, 10))

        # 3 Column readout
        col1 = tk.Frame(summary_frame, bg='#ffffff', padx=15, pady=10)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        col2 = tk.Frame(summary_frame, bg='#ffffff', padx=15, pady=10)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        col3 = tk.Frame(summary_frame, bg='#ffffff', padx=15, pady=10)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Labels for Col 1: Pathology
        tk.Label(col1, text="PREDICTED PATHOLOGY", font=('Segoe UI', 9, 'bold'), fg='#64748b', bg='#ffffff').pack(anchor=tk.W)
        self.res_path_lbl = tk.Label(col1, text="—", font=('Segoe UI', 14, 'bold'), fg='#0284c7', bg='#ffffff')
        self.res_path_lbl.pack(anchor=tk.W)
        self.res_path_sub = tk.Label(col1, text="Confidence: —", font=('Segoe UI', 9), fg='#475569', bg='#ffffff')
        self.res_path_sub.pack(anchor=tk.W)

        # Labels for Col 2: Wagner Depth
        tk.Label(col2, text="PREDICTED WAGNER GRADE", font=('Segoe UI', 9, 'bold'), fg='#64748b', bg='#ffffff').pack(anchor=tk.W)
        self.res_wag_lbl = tk.Label(col2, text="—", font=('Segoe UI', 14, 'bold'), fg='#0d9488', bg='#ffffff')
        self.res_wag_lbl.pack(anchor=tk.W)
        self.res_wag_sub = tk.Label(col2, text="Confidence: —", font=('Segoe UI', 9), fg='#475569', bg='#ffffff')
        self.res_wag_sub.pack(anchor=tk.W)

        # Labels for Col 3: University of Texas Matrix
        tk.Label(col3, text="UT WOUND STAGING", font=('Segoe UI', 9, 'bold'), fg='#64748b', bg='#ffffff').pack(anchor=tk.W)
        self.res_ut_lbl = tk.Label(col3, text="—", font=('Segoe UI', 14, 'bold'), fg='#7c3aed', bg='#ffffff')
        self.res_ut_lbl.pack(anchor=tk.W)
        self.res_ut_sub = tk.Label(col3, text="Stage: —", font=('Segoe UI', 9), fg='#475569', bg='#ffffff')
        self.res_ut_sub.pack(anchor=tk.W)

        # Tabbed Visualizations Notebook
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_cnn = ttk.Frame(self.notebook)
        self.tab_vit = ttk.Frame(self.notebook)
        self.tab_fused = ttk.Frame(self.notebook)
        self.tab_ig = ttk.Frame(self.notebook)
        self.tab_shap = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_overview, text=" 📊 Multi-Method Summary ")
        self.notebook.add(self.tab_cnn, text=" 🔍 CNN Grad-CAM ")
        self.notebook.add(self.tab_vit, text=" 🤖 Swin Transformer ")
        self.notebook.add(self.tab_fused, text=" ⚡ Dual-Branch Fused ")
        self.notebook.add(self.tab_ig, text=" 📈 Integrated Gradients ")
        self.notebook.add(self.tab_shap, text=" 🧩 KernelSHAP ")

        # Image view canvases
        self.canvases = {}
        for tab, key in [
            (self.tab_overview, 'overview'),
            (self.tab_cnn, 'cnn'),
            (self.tab_vit, 'vit'),
            (self.tab_fused, 'fused'),
            (self.tab_ig, 'ig'),
            (self.tab_shap, 'shap')
        ]:
            c = tk.Canvas(tab, bg='#1e293b')
            c.pack(fill=tk.BOTH, expand=True)
            self.canvases[key] = c

        self._show_welcome_in_canvases()

    def _show_welcome_in_canvases(self):
        for c in self.canvases.values():
            c.delete("all")
            c.create_text(
                450, 220, text="Please select a test image and click 'Run Diagnosis & XAI'",
                fill='#94a3b8', font=('Segoe UI', 12, 'italic'), justify=tk.CENTER
            )

    def _find_checkpoints(self):
        candidates = []
        for p in [
            "checkpoints_a1_full_balanced/best_checkpoint.pth",
            "checkpoints_a1_full/best_checkpoint.pth",
            "checkpoints_swin_tiny/best_checkpoint.pth"
        ]:
            if os.path.exists(p):
                candidates.append(p)

        found = glob.glob("checkpoints*/best_checkpoint.pth")
        for f in found:
            norm = os.path.normpath(f).replace('\\', '/')
            if norm not in candidates:
                candidates.append(norm)

        if candidates:
            self.ckpt_combo['values'] = candidates
            self.ckpt_combo.current(0)
            self._load_model_threaded(candidates[0])
        else:
            self.ckpt_status_lbl.config(text="No checkpoints found automatically. Please browse.")

    def _on_checkpoint_change(self):
        chosen = self.ckpt_var.get()
        if chosen:
            self._load_model_threaded(chosen)

    def _browse_custom_checkpoint(self):
        filename = filedialog.askopenfilename(
            title="Select DFU-SIAM v3 Checkpoint",
            filetypes=[("PyTorch Checkpoints", "*.pth *.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.ckpt_combo['values'] = list(self.ckpt_combo['values']) + [filename]
            self.ckpt_combo.set(filename)
            self._load_model_threaded(filename)

    def _load_model_threaded(self, path):
        self.ckpt_status_lbl.config(text=f"Loading: {os.path.basename(path)}...", foreground='#f59e0b')
        self.btn_run.config(state='disabled')

        def loader():
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                cfg = ckpt.get('config', {})
                arch = cfg.get('arch', 'full')
                mdl = DFUSiamV3(
                    arch=arch, pretrained=False,
                    shared_dim=cfg.get('shared_dim', None),
                    path_dim=cfg.get('path_dim', None),
                    depth_dim=cfg.get('depth_dim', None)
                ).to(self.device)
                mdl.load_state_dict(ckpt['model_state_dict'])
                mdl.eval()

                expl = XAIExplainer(mdl, device=self.device)
                self.model = mdl
                self.explainer = expl
                self.after(0, lambda: self._on_model_loaded_success(path, arch))
            except Exception as e:
                self.after(0, lambda: self._on_model_loaded_fail(str(e)))

        threading.Thread(target=loader, daemon=True).start()

    def _on_model_loaded_success(self, path, arch):
        self.ckpt_status_lbl.config(text=f"Loaded: {os.path.basename(path)} ({arch})", foreground='#16a34a')
        self.btn_run.config(state='normal')
        self.status_lbl.config(text="Model ready. Select an image to begin.")

    def _on_model_loaded_fail(self, err):
        self.ckpt_status_lbl.config(text=f"Error loading model", foreground='#dc2626')
        messagebox.showerror("Model Load Error", f"Failed to load checkpoint:\n{err}")

    def _browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Ulcer Test Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if filename:
            self.selected_img_path = filename
            try:
                img = Image.open(filename).convert("RGB")
                self.current_pil_img = img
                self.img_info_lbl.config(text=f"Resolution: {img.width}x{img.height} | {os.path.basename(filename)}")

                # Update thumbnail
                thumb = img.copy()
                thumb.thumbnail((210, 190), Image.Resampling.LANCZOS)
                self.thumb_photo = ImageTk.PhotoImage(thumb)
                self.thumb_canvas.delete("all")
                self.thumb_canvas.create_image(110, 100, image=self.thumb_photo)
                self.status_lbl.config(text="Image loaded. Click 'Run Diagnosis & XAI'.")
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Could not read image:\n{e}")

    def _start_analysis(self):
        if self.model is None or self.explainer is None:
            messagebox.showwarning("Model Not Ready", "Please load a valid model checkpoint first.")
            return

        if self.current_pil_img is None:
            messagebox.showwarning("No Image", "Please select an ulcer image to evaluate.")
            return

        self.btn_run.config(state='disabled')
        self.btn_select_img.config(state='disabled')
        self.progress_bar.pack(fill=tk.X, pady=(4, 4))
        self.progress_bar.start(10)
        self.status_lbl.config(text="Analyzing ulcer pathology, depth & computing XAI heatmaps...")

        target = self.target_var.get()
        method = self.method_var.get()
        pil_img = self.current_pil_img.copy()

        def worker():
            try:
                transform = get_dfu_transforms(is_train=False, img_size=(224, 224))
                results, meta = self.explainer.explain_image(
                    pil_img=pil_img,
                    transform=transform,
                    target_type=target,
                    target_id='predicted',
                    method=method
                )
                self.after(0, lambda: self._on_analysis_complete(pil_img, results, meta))
            except Exception as e:
                self.after(0, lambda: self._on_analysis_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analysis_complete(self, pil_img, results, meta):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run.config(state='normal')
        self.btn_select_img.config(state='normal')
        self.status_lbl.config(text=f"Completed in {meta.get('runtime_sec', 0.0):.2f}s!")

        self.cached_results = results
        self.cached_meta = meta

        # 1. Update Diagnostics Summary Cards
        p_name = meta['pred_pathology_name'].upper()
        p_prob = max(meta['pathology_probs']) * 100
        self.res_path_lbl.config(text=p_name)
        self.res_path_sub.config(text=f"Confidence: {p_prob:.1f}%")

        w_name = meta['pred_wagner_name'].upper()
        w_prob = max(meta['wagner_probs']) * 100
        self.res_wag_lbl.config(text=w_name)
        self.res_wag_sub.config(text=f"Confidence: {w_prob:.1f}%")

        # Exploratory UT Matrix calculation
        stage_letters = ['A', 'B', 'C', 'D']
        path_idx = meta['pred_pathology_id']
        depth_idx = meta['pred_wagner_grade']
        ut_str = f"Stage {stage_letters[path_idx]} (Depth {min(depth_idx, 3)})"
        ut_prob = (p_prob * w_prob) / 100.0
        self.res_ut_lbl.config(text=ut_str)
        self.res_ut_sub.config(text=f"Joint Prob: {ut_prob:.1f}%")

        # 2. Render Overlays into Canvases
        self.photo_references.clear()

        # Canvas Overview
        temp_sum_path = os.path.join("outputs", "temp_gui_summary.png")
        os.makedirs("outputs", exist_ok=True)
        self.explainer._create_summary_figure(pil_img, results, meta, temp_sum_path)
        if os.path.exists(temp_sum_path):
            self._render_image_on_canvas('overview', Image.open(temp_sum_path))

        # CNN Grad-CAM
        if 'gradcam' in results:
            cnn_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['cnn_heatmap'])
            self._render_image_on_canvas('cnn', Image.fromarray(cnn_ov), title="CNN Grad-CAM (EfficientNetV2 conv_head)")

            vit_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['vit_heatmap'])
            self._render_image_on_canvas('vit', Image.fromarray(vit_ov), title="Swin Transformer Attention (norm1)")

            fused_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['fused_heatmap'])
            self._render_image_on_canvas('fused', Image.fromarray(fused_ov), title="Dual-Branch Fused Explanation")

        if 'transformer' in results:
            t_ov, _ = overlay_heatmap_on_image(pil_img, results['transformer']['transformer_heatmap'])
            self._render_image_on_canvas('vit', Image.fromarray(t_ov), title="Swin-Tiny Token Attribution")

        if 'integrated_gradients' in results:
            ig_ov, _ = overlay_heatmap_on_image(pil_img, results['integrated_gradients']['ig_heatmap'])
            self._render_image_on_canvas('ig', Image.fromarray(ig_ov), title="Integrated Gradients Path Attribution")

        if 'shap' in results:
            shap_ov, _ = overlay_heatmap_on_image(pil_img, results['shap']['shap_heatmap'])
            self._render_image_on_canvas('shap', Image.fromarray(shap_ov), title="KernelSHAP Superpixel Attribution")

    def _render_image_on_canvas(self, key, img_obj, title=None):
        canvas = self.canvases.get(key)
        if not canvas:
            return

        canvas.delete("all")
        c_w = canvas.winfo_width() or 800
        c_h = canvas.winfo_height() or 500

        # Scale image to fit canvas preserving aspect ratio
        display_img = img_obj.copy()
        display_img.thumbnail((c_w - 40, c_h - 40), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(display_img)
        self.photo_references[key] = photo

        x = c_w // 2
        y = c_h // 2
        canvas.create_image(x, y, image=photo)

        if title:
            canvas.create_text(x, 20, text=title, fill='#38bdf8', font=('Segoe UI', 11, 'bold'))

    def _on_analysis_error(self, err):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_run.config(state='normal')
        self.btn_select_img.config(state='normal')
        self.status_lbl.config(text="Analysis failed.")
        messagebox.showerror("Analysis Error", f"An error occurred during inference:\n{err}")


if __name__ == "__main__":
    app = DFUSiamApp()
    app.mainloop()
