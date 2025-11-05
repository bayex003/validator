from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io, numpy as np
from PIL import Image

app = FastAPI(title="My Baby Gender Predictor Validator")

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img)

        # --- Lightweight validation ---
        gray_ratio = float(np.std(arr.mean(axis=2)) / 128)
        colorfulness = float(np.mean(np.abs(arr[..., 0] - arr[..., 1])) / 255)
        height, width = arr.shape[:2]

        if width < 200 or height < 200:
            label, score = "blurry", 0.5
        elif colorfulness > 0.25:
            label, score = "not_ultrasound", 0.6
        elif gray_ratio < 0.2:
            label, score = "ultrasound_front_back", 0.7
        else:
            label, score = "ultrasound_side", 0.9

        return {"label": label, "score": score}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
