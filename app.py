import io
import json
import traceback
import logging
from typing import Optional, List, Dict
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, WebSocket, Depends, status
from pydantic import BaseModel
from core.stt import ASRPipeline


SAMPLE_RATE = 16000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

app = FastAPI()
pipeline = ASRPipeline()
results_storage: Dict[str, Dict] = {}


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
            end_stream = meta.get("end_stream")
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

            transcription = pipeline(audio, end_stream=end_stream)
            results_storage[audio_id] = transcription

            await websocket.send_json({
                "id": audio_id,
                **transcription
            })

    except Exception:
        logger.info(f"WebSocket closed:\n{traceback.format_exc()}")


@app.post("/reset", status_code=status.HTTP_200_OK)
async def reset_results():
    global results_storage
    results_storage.clear()
    logger.info("All results have been reset")

    return {
        "status": "ok",
        "message": "All results have been reset"
    }


@app.post("/results")
async def get_results():
    """Get all stored results"""
    print("gdf")
    logger.info(f"Retrieved {len(results_storage)} results")
    return {
        "results": results_storage
    }


# uvicorn app:app --reload --port 8000
