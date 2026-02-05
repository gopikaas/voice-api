from fastapi import FastAPI, Header, HTTPException
import base64
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()
API_KEY = "12345"

@app.post("/detect-voice")
async def detect_voice(payload: dict, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        audio_base64 = payload["audio_base64"]
        language = payload["language"]

        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name

        y, sr = librosa.load(path, sr=16000)
        os.remove(path)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        score = np.mean(mfcc)

        if score > 0:
            result = "AI-generated"
            confidence = 0.90
        else:
            result = "human"
            confidence = 0.85

        return {
            "status": "success",
            "prediction": result,
            "confidence": confidence,
            "language": language,
            "model_version": "v1"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
