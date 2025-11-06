from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import os, io
from typing import Tuple

app = FastAPI(title="Baby Predictor Validator")

allowed_origins = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "validator", "status": "running"}

def to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32)
    return arr.astype(np.float32)

def sobel_edges(gray: np.ndarray) -> np.ndarray:
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    try:
        from scipy.signal import convolve2d
        gx = convolve2d(gray, Kx, mode="same", boundary="symm")
        gy = convolve2d(gray, Ky, mode="same", boundary="symm")
    except Exception:
        gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    return mag

def edge_density_and_contrast(gray: np.ndarray) -> Tuple[float, float]:
    g = (gray - gray.min()) / max(1e-6, (gray.max() - gray.min()))
    mag = sobel_edges(g)
    thresh = np.percentile(mag, 85)
    edge_mask = (mag > thresh).astype(np.uint8)
    density = float(edge_mask.mean())
    contrast = float(g.std())
    return density, contrast

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png",".jpg",".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    arr = np.array(img)
    h, w, _ = arr.shape
    if h < 220 or w < 220:
        raise HTTPException(status_code=400, detail="Image too small. Please upload a clear scan.")

    gray = to_gray(arr)
    density, contrast = edge_density_and_contrast(gray)

    likely_ultrasound = (0.03 <= density <= 0.25) and (contrast >= 0.07)
    conf = max(0.1, min(0.95, 0.5 * ((density-0.03) / 0.22) + 0.5 * ((contrast-0.07) / 0.23)))
    conf = float(round(conf, 2))

    if likely_ultrasound:
        return {
            "status": "ok",
            "label": "ultrasound_side",
            "confidence": conf,
            "message": "Valid ultrasound side-profile image (heuristic)."
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Upload rejected. Please upload a clear grayscale ultrasound side profile (side view, not front/back)."
        )