import asyncio
import json
import librosa
import numpy as np
import websockets

# ================= CONFIG =================
WS_URL = "ws://localhost:8000/bot/stt"
WAV_PATH = "data/topic_vi.mp3"   
SAMPLE_RATE = 16000
CHUNK_SEC = 5
# =========================================


def load_wav_chunks(path: str, chunk_sec: int):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    chunk_samples = chunk_sec * SAMPLE_RATE
    total_samples = len(audio)

    for i in range(0, total_samples, chunk_samples):
        chunk = audio[i:i + chunk_samples]

        if len(chunk) == 0:
            continue

        chunk = np.clip(chunk, -1.0, 1.0)
        chunk = (chunk * 32767).astype(np.int16)

        yield chunk.tobytes()


# async def sender(ws):
#     audio_id = 0

#     for chunk_bytes in load_wav_chunks(WAV_PATH, CHUNK_SEC):
#         meta = {
#             "id": f"audio_{audio_id}",
#         }

#         print(f"[SEND] audio_{audio_id}")
#         await ws.send(json.dumps(meta))
#         await ws.send(chunk_bytes)

#         audio_id += 1
#         await asyncio.sleep(CHUNK_SEC)  

async def sender(ws):
    audio_id = 0

    for chunk_bytes in load_wav_chunks(WAV_PATH, CHUNK_SEC):
        samples = len(chunk_bytes) // 2
        duration = samples / SAMPLE_RATE

        meta = {
            "id": f"audio_{audio_id}",
            "sample_rate": SAMPLE_RATE,
            "samples": samples
        }

        print(f"[SEND] audio_{audio_id} ({duration:.2f}s)")

        await ws.send(json.dumps(meta))
        await ws.send(chunk_bytes)

        audio_id += 1
        await asyncio.sleep(duration)


async def receiver(ws):
    while True:
        try:
            msg = await ws.recv()
            print("[RECV]", msg)
        except websockets.ConnectionClosed:
            break


async def main():
    async with websockets.connect(
        WS_URL,
        max_size=50 * 1024 * 1024
    ) as ws:
        await asyncio.gather(
            sender(ws),
            receiver(ws)
        )


if __name__ == "__main__":
    asyncio.run(main())
