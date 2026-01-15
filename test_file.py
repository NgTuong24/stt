import asyncio
import json
import librosa
import numpy as np
import websockets
import io

WS_URL = "ws://localhost:8000/bot/stt"
SAMPLE_RATE = 16000


def process_audio_file_bytes(file_bytes: bytes) -> bytes:
    """
    Input : bytes của file audio (wav/mp3)
    Output: PCM16 bytes (mono, 16kHz)
    """
    buffer = io.BytesIO(file_bytes)
    audio, _ = librosa.load(
        buffer,
        sr=SAMPLE_RATE,
        mono=True
    )
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    return audio.tobytes()

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
    
def load_audio_bytes_by_path(path: str) -> bytes:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    return audio.tobytes()


async def test_ws():
    async with websockets.connect(
        WS_URL,
        max_size=50 * 1024 * 1024
    ) as ws:
        audio_id = "audio_001"

        meta = {
            "id": audio_id,
        }
        await ws.send(json.dumps(meta))

        audio_bytes = load_audio_bytes_by_path("data/topic_vi.mp3")
        await ws.send(audio_bytes)

        result = await ws.recv()
        print("RESULT:", result)
        
# async def test_ws():
#     async with websockets.connect(
#         WS_URL,
#         max_size=50 * 1024 * 1024
#     ) as ws:
#         audio_bytes = load_audio_bytes_by_path("quat0.75x.mp3")
#         # file_bytes = read_file_bytes("quat0.75x.mp3")
#         # audio_bytes = process_audio_file_bytes(file_bytes)
#         print("Sending audio...")
#         await ws.send(audio_bytes)

#         result = await ws.recv()
#         print("\n===== RESULT =====")
#         print(result)


if __name__ == "__main__":
    asyncio.run(test_ws())

