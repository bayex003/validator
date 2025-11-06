from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io, os

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(title="Baby Gender Predictor Validator")

# CORS
allowed_origins = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "validator", "status": "running"}

# ---------------------------------------------------------------------
# Image classifier (mock validator for ultrasound)
# ---------------------------------------------------------------------
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Validate extension
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG and PNG files are supported.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Convert to array for mock validation
    img_array = np.array(image)
    h, w, _ = img_array.shape

    # -----------------------------------------------------------------
    # Fake detection rules (placeholder for your future ML model)
    # -----------------------------------------------------------------
    # Reject if too small / blurry
    if h < 200 or w < 200:
        raise HTTPException(status_code=400, detail="Image too small. Please upload a clear scan.")

    # Dummy check: random-ish rule to simulate classification
    mean_pixel = img_array.mean()
    label = "ultrasound_side" if 90 < mean_pixel < 200 else "not_ultrasound"

    # Confidence simulation
    confidence = round(float(abs(mean_pixel - 128) / 128), 2)

    # Determine message
    if label != "ultrasound_side":
        raise HTTPException(status_code=400, detail="Please upload a clear side-profile baby scan.")

    # Return validation result
    return {
        "status": "ok",
        "label": label,
        "confidence": confidence,
        "message": "Valid ultrasound side-profile image.",
    }
