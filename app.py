import io
import json
import traceback
import logging
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

from fastapi import FastAPI, WebSocket, Depends
from pydantic import BaseModel

from core.stt import ASRPipeline


SAMPLE_RATE = 16000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

app = FastAPI()


def bytes2audio(buffer: bytes, **kwargs):
    audio_file = sf.SoundFile(io.BytesIO(buffer), **kwargs)
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, dtype=np.float32)
    return audio


class ASRAudioInput(BaseModel):
    channels: Optional[int] = 1
    endian: Optional[str] = "LITTLE"
    samplerate: Optional[int] = 16000
    subtype: Optional[str] = "PCM_16"
    format: Optional[str] = "RAW"
    min_clusters: Optional[int] = 1
    max_clusters: Optional[int] = 5
        
        
pipeline = ASRPipeline()


# =====================
# WebSocket endpoint
# =====================
@app.websocket("/bot/stt")
async def bot_stt(
    websocket: WebSocket,
    input_audio_format: ASRAudioInput = Depends(),
):
    await websocket.accept()
    logger.info("[/bot/stt] WebSocket connected")

    try:
        while True:
            meta_msg = await websocket.receive_text()
            meta = json.loads(meta_msg)

            audio_id = meta.get("id")
            if not audio_id:
                await websocket.send_json({"error": "missing id"})
                continue

            buffer = await websocket.receive_bytes()

            if not buffer:
                await websocket.send_json({
                    "id": audio_id,
                    "full_text": ""
                })
                continue
            
            audio = bytes2audio(
                buffer,
                **input_audio_format.model_dump(
                    exclude={"min_clusters", "max_clusters"}
                ),
            )

            transcription = pipeline(audio)

            await websocket.send_json({
                "id": audio_id,
                **transcription
            })

    except Exception:
        logger.info(f"WebSocket closed:\n{traceback.format_exc()}")


# uvicorn app:app --reload --port 8000
