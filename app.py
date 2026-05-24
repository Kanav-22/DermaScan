
import torch
import torch.nn as nn
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# ── MODEL DEFINITION ─────────────────────────────────────────
class DermaScanModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        layers   = list(backbone.features.children())
        for layer in layers[:int(0.6 * len(layers))]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


# ── LOAD MODEL ────────────────────────────────────────────────
device = torch.device("cpu")  # HuggingFace free tier = CPU
model  = DermaScanModel().to(device)
ckpt   = torch.load("dermascan_hf.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded — AUC: {ckpt['val_auc']:.4f}")


# ── TRANSFORMS ───────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ── GRAD-CAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, pil_image):
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True
        output = self.model(image_tensor)
        prob   = torch.sigmoid(output).item()
        self.model.zero_grad()
        output.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.clamp(cam, min=0)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_np     = cam.cpu().numpy()
        w, h       = pil_image.size
        cam_resized = cv2.resize(cam_np, (w, h))
        heatmap    = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap    = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig_np    = np.array(pil_image.resize((w, h)))
        overlay    = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)
        return overlay, prob

gradcam = GradCAM(model, model.features[-1])


# ── ABCD SCORING ─────────────────────────────────────────────
def compute_abcd_scores(pil_image):
    img_np     = np.array(pil_image.convert("RGB"))
    img_resized = cv2.resize(img_np, (224, 224))
    gray       = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    blur       = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask    = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel     = np.ones((5, 5), np.uint8)
    mask       = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask       = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    largest    = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    flip_h     = cv2.flip(clean_mask, 1)
    flip_v     = cv2.flip(clean_mask, 0)
    area       = clean_mask.sum() + 1e-6
    asymmetry  = min((cv2.absdiff(clean_mask, flip_h).sum() +
                      cv2.absdiff(clean_mask, flip_v).sum()) / (4 * area), 1.0)
    perimeter  = cv2.arcLength(largest, True)
    l_area     = cv2.contourArea(largest) + 1e-6
    circularity = (4 * np.pi * l_area) / (perimeter ** 2 + 1e-6)
    border     = float(np.clip(1.0 - circularity, 0, 1))
    lesion_pixels = img_resized[clean_mask == 255].reshape(-1, 3).astype(np.float32)
    if len(lesion_pixels) > 100:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(lesion_pixels, 6, None, criteria, 3,
                                    cv2.KMEANS_RANDOM_CENTERS)
        color_score = float(np.clip(np.std(centers, axis=0).mean() / 128.0, 0, 1))
    else:
        color_score = 0.0
    diameter_score = float(np.clip(l_area / (224 * 224), 0, 1))
    return {"A": round(asymmetry, 3), "B": round(border, 3),
            "C": round(color_score, 3), "D": round(diameter_score, 3)}


# ── MC DROPOUT ───────────────────────────────────────────────
def mc_dropout_uncertainty(image_tensor, n_passes=30):
    model.train()
    image_tensor = image_tensor.to(device)
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            probs.append(torch.sigmoid(model(image_tensor)).item())
    model.eval()
    probs = np.array(probs)
    return probs.mean(), probs.std()


# ── ABCD CHART ───────────────────────────────────────────────
def make_abcd_chart(scores):
    fig, ax = plt.subplots(figsize=(4, 3))
    labels  = ["A\nAsym", "B\nBorder", "C\nColor", "D\nDiam"]
    values  = [scores["A"], scores["B"], scores["C"], scores["D"]]
    colors  = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.3 else "#2ecc71"
               for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4)
    ax.set_title("ABCD Scores", fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return Image.open(buf)


# ── MAIN ANALYSIS FUNCTION ────────────────────────────────────
def analyze(image):
    if image is None:
        return None, None, "Please upload an image."

    pil_img = Image.fromarray(image).convert("RGB")
    tensor  = infer_transform(pil_img).unsqueeze(0)

    gradcam_overlay, prob = gradcam.generate(tensor.clone(), pil_img)
    abcd_scores           = compute_abcd_scores(pil_img)
    mean_p, unc           = mc_dropout_uncertainty(tensor.clone())

    pred  = "MELANOMA" if mean_p >= 0.5 else "BENIGN"
    conf  = "High" if unc < 0.05 else "Moderate" if unc < 0.15 else "Low"

    summary = f"""
**Prediction: {pred}**

P(melanoma): {mean_p:.3f}
Uncertainty: ±{unc:.3f}
Confidence:  {conf}

ABCD Scores:
  A (Asymmetry) : {abcd_scores["A"]:.3f}
  B (Border)    : {abcd_scores["B"]:.3f}
  C (Color)     : {abcd_scores["C"]:.3f}
  D (Diameter)  : {abcd_scores["D"]:.3f}

⚠ Not a medical diagnosis. Consult a dermatologist.
    """.strip()

    abcd_chart = make_abcd_chart(abcd_scores)
    return Image.fromarray(gradcam_overlay), abcd_chart, summary


# ── GRADIO INTERFACE ─────────────────────────────────────────
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(label="Upload skin lesion photo"),
    outputs=[
        gr.Image(label="Grad-CAM — Where the model looked"),
        gr.Image(label="ABCD Scores"),
        gr.Markdown(label="Analysis Result")
    ],
    title="DermaScan — AI Skin Lesion Analyzer",
    description=(
        "Upload a photo of a skin lesion. "
        "DermaScan uses EfficientNet B0 trained on HAM10000 (AUC 0.93) "
        "with Grad-CAM explainability and ABCD clinical scoring.\n\n"
        "**This is a screening tool only. Always consult a qualified dermatologist.**"
    ),
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
