"""
Unified Explainable AI (XAI) Explainer for DFU-SIAM v3.
Integrates:
- GradCAMExplainer (CNN & ViT branch heatmaps)
- TransformerExplainer (Token-level Swin attention importance)
- SHAPExplainer (Superpixel Shapley values)
- IntegratedGradientsExplainer (Path-integrated gradients)
- Publication-quality visualization and summary packaging
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from xai.base import (
    BaseExplainer, PATHOLOGY_CLASSES, WAGNER_GRADES,
    overlay_heatmap_on_image, normalize_attribution_map
)
from xai.gradcam import GradCAMExplainer
from xai.transformer import TransformerExplainer
from xai.integrated_gradients import IntegratedGradientsExplainer
from xai.shap_explainer import SHAPExplainer
from xai.evaluation import run_deletion_test, run_insertion_test


class XAIExplainer:
    """
    Unified manager for all DFU-SIAM v3 explainability tools.
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()

        self.gradcam = GradCAMExplainer(model, self.device)
        self.transformer = TransformerExplainer(model, self.device)
        self.ig = IntegratedGradientsExplainer(model, steps=25, baseline_type='black', device=self.device)
        self.shap = SHAPExplainer(model, grid_size=14, n_evals=100, device=self.device)

    def explain_image(
        self,
        pil_img,
        transform,
        target_type='pathology',
        target_id='predicted',
        method='all',
        output_dir=None,
        run_faithfulness=False,
        extra_meta=None
    ):
        """
        Executes XAI methods for a given image.
        Supports method in ['gradcam', 'transformer', 'integrated_gradients', 'shap', 'all'].
        Saves visual overlays, attribution maps, and explanation.json if output_dir is given.
        """
        self.model.eval()
        input_tensor = transform(pil_img).unsqueeze(0).to(self.device)

        results = {}
        # 1. Grad-CAM
        if method in ['gradcam', 'all']:
            results['gradcam'] = self.gradcam.explain(input_tensor, target_type=target_type, target_id=target_id)

        # 2. Transformer
        if method in ['transformer', 'all']:
            results['transformer'] = self.transformer.explain(input_tensor, target_type=target_type, target_id=target_id)

        # 3. Integrated Gradients
        if method in ['integrated_gradients', 'all']:
            results['integrated_gradients'] = self.ig.explain(input_tensor, target_type=target_type, target_id=target_id)

        # 4. SHAP
        if method in ['shap', 'all']:
            results['shap'] = self.shap.explain(input_tensor, target_type=target_type, target_id=target_id)

        # Extract shared metadata from the first executed method
        first_key = list(results.keys())[0]
        meta = results[first_key]['meta'].copy()
        meta['model_version'] = getattr(self.model, 'model_version', 'v3')
        meta['model_architecture'] = getattr(self.model, 'arch', 'full')
        meta['xai_method'] = method

        layers_used = {}
        if 'gradcam' in results:
            layers_used['cnn'] = 'model.backbone.cnn.conv_head'
            layers_used['vit'] = 'model.backbone.vit.layers[-1].blocks[-1].norm1'
            layers_used['fused'] = 'dual-branch symmetrical fusion (0.5 * CNN + 0.5 * ViT)'
        if 'transformer' in results:
            layers_used['transformer'] = 'model.backbone.vit.layers[-1].blocks[-1].norm1'
        if 'integrated_gradients' in results:
            layers_used['integrated_gradients'] = 'input_pixel_space'
        if 'shap' in results:
            layers_used['shap'] = f'superpixel_grid_{self.shap.grid_size}x{self.shap.grid_size}'
        meta['layers_used'] = layers_used

        meta['xai_configuration'] = {
            'ig_steps': self.ig.steps,
            'ig_baseline': self.ig.baseline_type,
            'shap_grid_size': self.shap.grid_size,
            'shap_n_evals': self.shap.n_evals,
        }

        if extra_meta:
            meta.update(extra_meta)

        # Run quantitative faithfulness if requested
        if run_faithfulness:
            ref_map = None
            if 'gradcam' in results:
                ref_map = results['gradcam']['cnn_heatmap']
            elif 'transformer' in results:
                ref_map = results['transformer']['transformer_heatmap']
            elif 'integrated_gradients' in results:
                ref_map = results['integrated_gradients']['ig_heatmap']

            if ref_map is not None:
                del_res = run_deletion_test(self.model, input_tensor, ref_map, target_type=target_type, target_id=target_id, device=self.device)
                ins_res = run_insertion_test(self.model, input_tensor, ref_map, target_type=target_type, target_id=target_id, device=self.device)
                meta['faithfulness_evaluation'] = {
                    'aopc_deletion_score': del_res['aopc_deletion_score'],
                    'insertion_auc': ins_res['insertion_auc']
                }

        # Save artifacts if output_dir specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pil_img.save(os.path.join(output_dir, "original.png"))

            def _save_raw(norm_map, filename):
                cmap = plt.get_cmap('jet')
                colored = (cmap(norm_map)[:, :, :3] * 255.0).astype(np.uint8)
                Image.fromarray(colored).save(os.path.join(output_dir, filename))

            # Save individual heatmaps and overlays
            if 'gradcam' in results:
                cnn_ov, cnn_raw = overlay_heatmap_on_image(pil_img, results['gradcam']['cnn_heatmap'])
                vit_ov, vit_raw = overlay_heatmap_on_image(pil_img, results['gradcam']['vit_heatmap'])
                fused_ov, fused_raw = overlay_heatmap_on_image(pil_img, results['gradcam']['fused_heatmap'])

                Image.fromarray(cnn_ov).save(os.path.join(output_dir, "gradcam_cnn.png"))
                Image.fromarray(vit_ov).save(os.path.join(output_dir, "gradcam_vit.png"))
                Image.fromarray(fused_ov).save(os.path.join(output_dir, "gradcam_fused.png"))
                _save_raw(cnn_raw, "gradcam_cnn_raw.png")

            if 'transformer' in results:
                trans_ov, trans_raw = overlay_heatmap_on_image(pil_img, results['transformer']['transformer_heatmap'])
                Image.fromarray(trans_ov).save(os.path.join(output_dir, "transformer_attribution.png"))
                _save_raw(trans_raw, "transformer_raw.png")

            if 'integrated_gradients' in results:
                ig_ov, ig_raw = overlay_heatmap_on_image(pil_img, results['integrated_gradients']['ig_heatmap'])
                Image.fromarray(ig_ov).save(os.path.join(output_dir, "integrated_gradients.png"))
                _save_raw(ig_raw, "integrated_gradients_raw.png")

            if 'shap' in results:
                shap_ov, shap_raw = overlay_heatmap_on_image(pil_img, results['shap']['shap_heatmap'])
                Image.fromarray(shap_ov).save(os.path.join(output_dir, "shap.png"))
                _save_raw(shap_raw, "shap_raw.png")

            # Create side-by-side publication comparison figure
            self._create_summary_figure(pil_img, results, meta, os.path.join(output_dir, "combined_summary.png"))

            # Export JSON metadata
            with open(os.path.join(output_dir, "explanation.json"), "w") as f:
                json.dump(meta, f, indent=4)

        return results, meta

    def _create_summary_figure(self, pil_img, results, meta, save_path):
        """Builds multi-panel publication summary comparison."""
        panels = [('Original Image', np.array(pil_img))]

        if 'gradcam' in results:
            cnn_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['cnn_heatmap'])
            panels.append(('CNN Grad-CAM\n(EfficientNetV2)', cnn_ov))
            fused_ov, _ = overlay_heatmap_on_image(pil_img, results['gradcam']['fused_heatmap'])
            panels.append(('Fused Attribution\n(Dual-Branch)', fused_ov))

        if 'transformer' in results:
            trans_ov, _ = overlay_heatmap_on_image(pil_img, results['transformer']['transformer_heatmap'])
            panels.append(('Transformer Attribution\n(Swin-Tiny Window Attention)', trans_ov))

        if 'integrated_gradients' in results:
            ig_ov, _ = overlay_heatmap_on_image(pil_img, results['integrated_gradients']['ig_heatmap'])
            panels.append(('Integrated Gradients\n(Path Attribution)', ig_ov))

        if 'shap' in results:
            shap_ov, _ = overlay_heatmap_on_image(pil_img, results['shap']['shap_heatmap'])
            panels.append(('SHAP Partition\n(Superpixel Importance)', shap_ov))

        num_cols = len(panels)
        fig, axes = plt.subplots(1, num_cols, figsize=(3.8 * num_cols, 4.2), dpi=250)
        if num_cols == 1:
            axes = [axes]

        for ax, (title, img_arr) in zip(axes, panels):
            ax.imshow(img_arr)
            ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
            ax.axis('off')

        target_str = f"Target: {meta.get('target_type', 'Pathology').upper()} -> '{meta.get('target_name', 'Predicted')}' (Score: {meta.get('target_score', 0.0):.3f})"
        pred_summary = f"Predicted Pathology: {meta.get('pred_pathology_name', 'N/A').upper()} | Predicted Wagner: {meta.get('pred_wagner_name', 'N/A')}"
        fig.suptitle(f"DFU-SIAM v3 Model Attribution Explanations\n{target_str} | {pred_summary}", fontsize=11, y=1.04)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
