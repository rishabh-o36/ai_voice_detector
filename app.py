from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from audio_utils import decode_audio
from model import predict

API_KEY = "YOUR_SECRET_KEY"

app = FastAPI(title="AI Generated Voice Detection API")

class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/detect")
def detect_voice(
    request: AudioRequest,
    authorization: str = Header(None)
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    waveform = decode_audio(request.audio_base64)
    label, confidence = predict(waveform)

    return {
        "classification": label,
        "confidence": round(confidence, 4)
    }
