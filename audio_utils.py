import base64
import io
import librosa
import torch
import numpy as np

TARGET_SR = 16000

def decode_audio(base64_audio: str):
    audio_bytes = base64.b64decode(base64_audio)
    audio_buffer = io.BytesIO(audio_bytes)

    waveform, sr = librosa.load(audio_buffer, sr=TARGET_SR, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0)

    return waveform
