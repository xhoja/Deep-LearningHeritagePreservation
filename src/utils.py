import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
import torch
import torch.nn as nn
import cv2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> dict:
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds, average='binary'),
        'auc_roc':  roc_auc_score(labels, probs[:, 1]),
    }


def print_classification_report(labels, preds, class_names):
    print(classification_report(labels, preds, target_names=class_names))


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['val_loss'],   label='val')
    axes[0].set_title('Loss'); axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='train')
    axes[1].plot(epochs, history['val_acc'],   label='val')
    axes[1].set_title('Accuracy'); axes[1].legend()

    axes[2].plot(epochs, history['val_f1'],      label='F1')
    axes[2].plot(epochs, history['val_auc_roc'], label='AUC-ROC')
    axes[2].set_title('Val Metrics'); axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def save_history(history: dict, output_dir: str):
    path = Path(output_dir) / 'history.json'
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay(image_rgb: np.ndarray, cam: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return (alpha * heatmap + (1 - alpha) * image_rgb).astype(np.uint8)
