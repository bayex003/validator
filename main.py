# validator/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import os, io
from typing import Tuple, Dict

app = FastAPI(title="Baby Predictor Validator (strict)")

# ✅ Your explicit CORS config preserved as-is (same origins + credentials + exposed headers)
ALLOWED_ORIGINS = [
    "https://mybabygenderpredictor.com",
    "https://www.mybabygenderpredictor.com",
    "https://mybabygenderpredictor-webapp.vercel.app",
    "http://localhost:3000",  # for local testing
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,      # keep your original setting
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "validator", "status": "running"}

# -------------------------
# Helpers / feature checks
# -------------------------

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR uint8"""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def to_gray(arr: np.ndarray) -> np.ndarray:
    """Legacy helper kept for compatibility (not used in the new checks)."""
    if arr.ndim == 3:
        return (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32)
    return arr.astype(np.float32)

def grayscale_ratio(bgr: np.ndarray) -> float:
    """How close the image is to grayscale: 1.0 = very gray, 0.0 = colorful."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    # scale saturation into [0..1], invert (higher = more grayscale)
    return float(1.0 - np.clip(sat.mean() * 2.0, 0, 1))

def blur_score(gray_u8: np.ndarray) -> float:
    """Variance of Laplacian; lower = blurrier."""
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

def canny_edge_density(gray_u8: np.ndarray) -> float:
    """Fraction of strong edges from Canny relative to pixels."""
    med = np.median(gray_u8)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray_u8, lower, upper)
    return float(np.mean(edges > 0))

def midtone_concentration(gray_u8: np.ndarray) -> float:
    """Ultrasound images usually have lots of mid-tones instead of pure black/white."""
    g = gray_u8.astype(np.float32) / 255.0
    return float(np.mean((g > 0.2) & (g < 0.8)))

def local_variance(gray_u8: np.ndarray) -> float:
    """Proxy for ultrasound 'speckle' texture (variance of residual)."""
    g = gray_u8.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    resid = g - blur
    return float(np.var(resid))

def orientation_anisotropy(gray_u8: np.ndarray) -> float:
    """Side-profile tends to show a preferred edge direction (anisotropy)."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.arctan2(gy, gx)
    mag = np.sqrt(gx * gx + gy * gy)

    bins = 36
    hist, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-6)
    # concentration = peak / (entropy-ish spread)
    concentration = float(hist.max() / (np.exp(-np.sum(hist * np.log(hist + 1e-8))) + 1e-6))
    return concentration

def strict_ultrasound_gate(bgr: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
    """
    Strong, explainable gate for ultrasound side-profile.
    Returns (ok, reason, features).
    """
    h, w, _ = bgr.shape
    if h < 320 or w < 320:
        return False, "Image too small. Please upload a clear scan (≥ 320×320).", {}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    feats = {
        "grayscale_ratio": grayscale_ratio(bgr),        # expect high (grayscale)
        "blur":            blur_score(gray),            # expect above threshold
        "edge_density":    canny_edge_density(gray),    # in plausible band
        "midtone":         midtone_concentration(gray), # mid tones present
        "local_var":       local_variance(gray),        # speckle/texture
        "anisotropy":      orientation_anisotropy(gray) # side-profile directionality
    }

    # Strict but sensible thresholds (tune if you see valid rejects):
    if feats["grayscale_ratio"] < 0.75:
        return False, "Image appears too colorful; ultrasound scans are grayscale.", feats

    if feats["blur"] < 60.0:
        return False, "Image looks too blurry for analysis. Please upload a clearer scan.", feats

    if not (0.02 <= feats["edge_density"] <= 0.22):
        return False, "Edge structure looks implausible for an ultrasound side-profile.", feats

    if feats["midtone"] < 0.45:
        return False, "Contrast distribution does not resemble an ultrasound image.", feats

    if feats["local_var"] < 0.0015:
        return False, "Texture pattern is too smooth; ultrasound speckle not detected.", feats

    if feats["anisotropy"] < 1.4:
        return False, "Edge orientation doesn’t look like a side-profile view.", feats

    return True, "Valid ultrasound side-profile (heuristic).", feats

# -------------------------
# Endpoint (same contract)
# -------------------------

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # keep your file-type guard
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported.")

    # decode as before, but pass to the strict gate
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    bgr = pil_to_bgr(img)
    ok, reason, feats = strict_ultrasound_gate(bgr)

    if not ok:
        # ❌ Reject with clear reason (no boy/girl guess for random images)
        raise HTTPException(status_code=400, detail=reason)

    # ✅ If it looks like a proper ultrasound side-profile, return success.
    # Provide a confidence proxy derived from features to keep your UI consistent.
    conf_parts = []
    conf_parts.append(np.clip((feats["grayscale_ratio"] - 0.70) / 0.30, 0, 1))
    conf_parts.append(np.clip((feats["blur"] - 60.0) / 140.0, 0, 1))
    conf_parts.append(np.clip((feats["edge_density"] - 0.02) / (0.22 - 0.02), 0, 1))
    conf_parts.append(np.clip((feats["midtone"] - 0.45) / 0.40, 0, 1))
    conf_parts.append(np.clip((feats["local_var"] - 0.0015) / 0.01, 0, 1))
    conf_parts.append(np.clip((feats["anisotropy"] - 1.4) / 1.0, 0, 1))
    confidence = float(np.mean(conf_parts))
    confidence = round(0.5 + 0.5 * confidence, 2)  # ~0.5..1.0

    return {
        "status": "ok",
        "label": "ultrasound_side",
        "confidence": confidence,
        "message": reason,
        # "features": feats,  # uncomment if you want to inspect/tune
    }