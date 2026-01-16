import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import timm
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['COVID-19', 'Pneumonia', 'Normal']

# paths (edit if needed)
EFFICIENTNET_PATH = r"D:\EDAI\Abnormality Detection Model\covidqu_model.pth"
DENSENET_PATH = r"D:\EDAI\Abnormality Detection Model\densenet.pth"
INCEPTION_PATH = r"D:\EDAI\Abnormality Detection Model\inceptionResNetV2.pth"   # rename your file to this or change here
RESNET18_PATH = r"D:\EDAI\Abnormality Detection Model\lung_segementation.pth"
VGG_PATH = r"D:\EDAI\Abnormality Detection Model\vgg16_covid_final_best_model.pth"

# segmentation is Unet++ saved via save_pretrained to HF hub
SEGMENTATION_REPO = "saketpatayeet/infection-segmentation"

# ensemble weights in order: [Eff, ResNet18, DenseNet121, VGG16, InceptionResNetV2]
ENSEMBLE_WEIGHTS = [0.3, 0.1, 0.4, 0.1, 0.1]

# image sizes
CLF_SIZE = 224          # works for InceptionResNetV2 & others
SEG_SIZE = 256          # you can change if you trained with other size


# -------------------------------------------------
# PREPROCESSING (NO FLIPS)
# -------------------------------------------------
clf_transform = transforms.Compose([
    transforms.Resize((CLF_SIZE, CLF_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

vis_transform = transforms.Compose([
    transforms.Resize((CLF_SIZE, CLF_SIZE)),
])

seg_transform = transforms.Compose([
    transforms.Resize((SEG_SIZE, SEG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------------------------------
# HELPER: LOAD STATE
# -------------------------------------------------
def load_state(path):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    return ckpt


# -------------------------------------------------
# MODELS: ENSEMBLE
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_classification_models():
    models_list = []

    # 1) EfficientNet-B0
    eff = EfficientNet.from_name('efficientnet-b0')
    eff._fc = nn.Linear(eff._fc.in_features, len(CLASS_NAMES))
    eff.load_state_dict(load_state(EFFICIENTNET_PATH))
    eff.to(DEVICE).eval()
    models_list.append(("EfficientNet-B0", eff))

    # 2) ResNet18
    resnet18 = models.resnet18(weights=None)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, len(CLASS_NAMES))
    resnet18.load_state_dict(load_state(RESNET18_PATH))
    resnet18.to(DEVICE).eval()
    models_list.append(("ResNet18", resnet18))

    # 3) DenseNet121
    densenet = models.densenet121(weights=None)
    densenet.classifier = nn.Linear(densenet.classifier.in_features,
                                    len(CLASS_NAMES))
    densenet.load_state_dict(load_state(DENSENET_PATH))
    densenet.to(DEVICE).eval()
    models_list.append(("DenseNet121", densenet))

    # 4) VGG16 with custom classifier
    vgg_ckpt = torch.load(VGG_PATH, map_location=DEVICE)
    state_dict = vgg_ckpt["model_state_dict"]
    state_dict = {k.replace("vgg16.", ""): v for k, v in state_dict.items()}

    vgg = models.vgg16(weights=None)
    vgg.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1024, len(CLASS_NAMES)),
    )
    vgg.load_state_dict(state_dict)
    vgg.to(DEVICE).eval()
    models_list.append(("VGG16", vgg))

    # 5) InceptionResNetV2 from timm
    inception = timm.create_model(
        "inception_resnet_v2", pretrained=False, num_classes=len(CLASS_NAMES)
    )
    inception.load_state_dict(load_state(INCEPTION_PATH))
    inception.to(DEVICE).eval()
    models_list.append(("InceptionResNetV2", inception))

    return models_list


# -------------------------------------------------
# MODELS: SEGMENTATION (Unet++)
# -------------------------------------------------
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load
@st.cache_resource(show_spinner=True)
def load_segmentation_model():
    # download config + weights
    config_path = hf_hub_download(SEGMENTATION_REPO, "config.json")
    weights_path = hf_hub_download(SEGMENTATION_REPO, "model.safetensors")

    # load config
    import json
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Rebuild architecture EXACTLY as saved
    model = smp.UnetPlusPlus(
        encoder_name=cfg["encoder_name"],         # e.g., "resnet18"
        encoder_weights=None,                     # you trained from scratch
        in_channels=cfg["in_channels"],           # typically 3
        classes=cfg["classes"],                   # typically 1
    )

    # Load safetensors
    state_dict = safe_load(weights_path)
    model.load_state_dict(state_dict)

    model.to(DEVICE).eval()
    return model


# -------------------------------------------------
# GRAD-CAM (INCEPTION RESNET V2)
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        # backward_hook is deprecated but fine for this use
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # input_tensor: (1, C, H, W)
        input_tensor = input_tensor.to(DEVICE)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap) + 1e-8
        return heatmap.numpy()


def create_gradcam_image(pil_img, heatmap):
    """Overlay heatmap on resized original image and return a side-by-side PIL image."""
    img_vis = vis_transform(pil_img).convert("RGB")
    img_np = np.array(img_vis).astype(np.float32) / 255.0

    # resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_color = heatmap_color[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB

    cam = heatmap_color + img_np
    cam = cam / cam.max()

    # side-by-side
    left = (img_np * 255).astype(np.uint8)
    right = (cam * 255).astype(np.uint8)
    combo = np.concatenate([left, right], axis=1)
    return Image.fromarray(combo)


# -------------------------------------------------
# ENSEMBLE PREDICTION
# -------------------------------------------------
def ensemble_predict(models_list, weights, input_tensor):
    probs_all = []
    per_model_probs = {}

    with torch.no_grad():
        for (name, model) in models_list:
            logits = model(input_tensor.to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            per_model_probs[name] = probs
            probs_all.append(probs)

    probs_all = np.stack(probs_all, axis=0)
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()

    ensemble_probs = (probs_all * weights_arr[:, None]).sum(axis=0)
    return ensemble_probs, per_model_probs


# -------------------------------------------------
# SEGMENTATION PREDICTION + OVERLAY
# -------------------------------------------------
def segment_and_overlay(seg_model, pil_img):
    img_vis = vis_transform(pil_img).convert("RGB")
    img_np = np.array(img_vis).astype(np.float32) / 255.0

    x = seg_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_pred = seg_model(x)
    # assume binary: (1,1,H,W) or (1,H,W)
    if mask_pred.ndim == 4:
        mask_pred = mask_pred[0, 0]
    else:
        mask_pred = mask_pred[0]

    mask_pred = torch.sigmoid(mask_pred)
    mask = mask_pred.cpu().numpy()
    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    mask_color = np.zeros_like(img_np)
    mask_color[:, :, 0] = mask  # red channel
    alpha = 0.5
    overlay = img_np * (1 - alpha) + mask_color * alpha
    overlay = (overlay / overlay.max()) if overlay.max() > 0 else overlay

    combo = np.concatenate(
        [(img_np * 255).astype(np.uint8),
         (overlay * 255).astype(np.uint8)],
        axis=1,
    )
    return Image.fromarray(combo)


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="Chest X-Ray Ensemble (COVID / Pneumonia / Normal)",
                   page_icon="🫁",
                   layout="wide")

st.title("🫁 Chest X-Ray Diagnosis with Ensemble + Grad-CAM + Segmentation")


with st.sidebar:
    st.subheader("Models")
    st.write("- EfficientNet-B0")
    st.write("- ResNet18")
    st.write("- DenseNet121")
    st.write("- VGG16 (custom head)")
    st.write("- InceptionResNetV2 (Grad-CAM source)")
    st.write("- Unet++ (ResNet18 encoder) for infection masks")

    st.markdown("---")
    st.info(f"Running on: **{DEVICE}**")

# load models once
with st.spinner("Loading classification models..."):
    ensemble_models = load_classification_models()
with st.spinner("Loading segmentation model..."):
    seg_model = load_segmentation_model()

uploaded = st.file_uploader("Upload a chest X-ray image (PNG / JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Running ensemble prediction..."):
            # classification tensor
            x_clf = clf_transform(pil_img).unsqueeze(0)

            # ensemble prediction
            ensemble_probs, per_model_probs = ensemble_predict(
                ensemble_models, ENSEMBLE_WEIGHTS, x_clf
            )

            pred_idx = int(np.argmax(ensemble_probs))
            pred_class = CLASS_NAMES[pred_idx]
            pred_conf = float(ensemble_probs[pred_idx])

        st.subheader("🧠 Ensemble Prediction")
        st.markdown(f"**Prediction:** `{pred_class}`  \n**Confidence:** `{pred_conf*100:.2f}%`")

        # show per-class probabilities
        cols = st.columns(len(CLASS_NAMES))
        for i, cname in enumerate(CLASS_NAMES):
            with cols[i]:
                st.metric(cname, f"{ensemble_probs[i]*100:.1f}%")

        # per-model table
        with st.expander("Show individual model probabilities"):
            for name, probs in per_model_probs.items():
                st.write(f"**{name}**")
                st.write(
                    {c: f"{p*100:.2f}%" for c, p in zip(CLASS_NAMES, probs)}
                )

        # ---------------- Grad-CAM ----------------
        with st.spinner("Generating Grad-CAM (InceptionResNetV2)..."):
            # pick inception model from list
            inception = dict(ensemble_models)["InceptionResNetV2"]
            # target layer – last conv; adjust if needed
            target_layer = inception.conv2d_7b
            grad_cam = GradCAM(inception, target_layer)
            heatmap = grad_cam.generate(x_clf, class_idx=pred_idx)
            gradcam_img = create_gradcam_image(pil_img, heatmap)

        st.subheader("🔥 Grad-CAM (InceptionResNetV2)")
        st.image(gradcam_img, caption="Original | Grad-CAM", use_column_width=True)

        buf_gc = io.BytesIO()
        gradcam_img.save(buf_gc, format="PNG")
        st.download_button(
            "Download Grad-CAM image",
            data=buf_gc.getvalue(),
            file_name="gradcam.png",
            mime="image/png",
        )

        # ---------------- Segmentation ----------------
        if pred_class == "COVID-19":
            with st.spinner("Running Unet++ infection segmentation..."):
                seg_img = segment_and_overlay(seg_model, pil_img)

            st.subheader("🩻 Infection Segmentation (Unet++)")
            st.image(seg_img, caption="Original | Segmentation overlay",
                     use_column_width=True)

            buf_seg = io.BytesIO()
            seg_img.save(buf_seg, format="PNG")
            st.download_button(
                "Download segmentation image",
                data=buf_seg.getvalue(),
                file_name="segmentation.png",
                mime="image/png",
            )
        else:
            st.info("Segmentation is only shown for **COVID-19** predictions.")
else:
    st.info("Upload an X-ray image to begin.")
