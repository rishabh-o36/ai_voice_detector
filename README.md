### AI‑Generated Voice Detection API
## Overview
A REST API that detects whether an input voice sample is AI‑generated or Human.
The system supports Tamil, English, Hindi, Malayalam, and Telugu and returns a confidence score.
--
## Approach
Accept Base64‑encoded MP3 audio

Decode and resample to 16 kHz

Run inference using a Wav2Vec2‑based deepfake voice detection model

Return classification + confidence in JSON
--
## Model
Model: koyelog/deepfake-voice-detector-sota

Type: Audio classification (Wav2Vec2)

Trained on: ASVspoof, WaveFake

Language‑agnostic, works well for Indian languages

API Specification
Endpoint
POST /detect
Headers
Authorization: Bearer <API_KEY>
Request
{
  "audio_base64": "<Base64 MP3>"
}
Response
{
  "classification": "AI_GENERATED",
  "confidence": 0.97
}
--
## Tech Stack
API: FastAPI

ML: PyTorch, Transformers

Audio: Librosa, Torchaudio
--
## Compliance
- No hard‑coding
- No external detection APIs
- JSON response with confidence
- Public REST endpoint
--
## Conclusion
This solution provides a robust, explainable, and scalable approach to detecting AI‑generated speech and fully complies with the buildathon requirements.