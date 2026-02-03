from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from audio_utils import decode_audio
from model import predict

API_KEY = "AIzaSyBelRXAJMPZpmdDxgohggt2TXr9E9y82kY"

app = FastAPI(title="Voice Classifier API")


class AudioRequest(BaseModel):
    audio_base64: str


@app.post("/detect")
def detect_voice(request: AudioRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        waveform = decode_audio(request.audio_base64)

        if waveform.shape[1] < 16000:
            raise HTTPException(
                status_code=400,
                detail="Audio too short (minimum 1 second required)"
            )

        label, confidence = predict(waveform)

        return {
            "classification": label,
            "confidence": round(confidence, 4)
        }

    except HTTPException:
        raise

    except Exception as e:
        print("INFERENCE ERROR:", e)
        raise HTTPException(status_code=500, detail="Internal inference error")
