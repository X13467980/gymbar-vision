# src/api/main.py
import os, io
from typing import Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from PIL import Image

API_KEY = os.getenv("VISION_API_KEY", "dev-key")

app = FastAPI(title="Gymba+R Vision API", version="v0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "version": "v0.1.0"}

def preprocess(image_bytes: bytes, size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)  # (1, H, W, 3)
    return x

@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    img_bytes = await file.read()
    _ = preprocess(img_bytes)  # 前処理のみ実施

    # ダミー：既定クラスへディリクレ分布で確率を割り振り
    labels = ["lat_pulldown", "chest_press", "leg_press", "smith_machine"]
    probs = np.random.dirichlet(np.ones(len(labels)), size=1)[0]
    idx = int(np.argmax(probs))
    return {"label": labels[idx], "prob": float(probs[idx]), "model_version": "dummy-v0.1.0"}