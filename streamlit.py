# app.py
import streamlit as st
import asyncio
import json
import librosa
import numpy as np
import websockets
import tempfile
import time

# ================= CONFIG =================
WS_URL = "ws://localhost:8000/bot/stt"
SAMPLE_RATE = 16000
# =========================================


# ================= AUDIO UTILS =================
def load_audio_chunks(path, chunk_sec):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    chunk_samples = int(chunk_sec * SAMPLE_RATE)

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) == 0:
            continue

        pcm16 = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)

        yield pcm16.tobytes(), len(chunk) / SAMPLE_RATE


# ================= STREAMING =================
async def stream_audio(ws_url, wav_path, chunk_sec, log_cb):
    async with websockets.connect(
        ws_url,
        max_size=50 * 1024 * 1024
    ) as ws:

        async def receiver():
            while True:
                try:
                    msg = await ws.recv()
                    msg = json.loads(msg)
                    log_cb(f"[RECV] {msg["id"]}: {msg["full_text"]}")
                except:
                    break

        async def sender():
            ind = 0
            audio_id = 0
            for chunk_bytes, duration in load_audio_chunks(wav_path, chunk_sec):
                ind += 1
                # TEST :
                end_stream = False
                if ind == 14:
                    end_stream = True
                #
                meta = {
                    "id": f"audio_{audio_id}",
                    "sample_rate": SAMPLE_RATE,
                    "bytes": len(chunk_bytes),
                    "end_stream": end_stream
                }

                log_cb(f"[SEND] audio_{audio_id} ({duration:.2f}s)")
                await ws.send(json.dumps(meta))
                await ws.send(chunk_bytes)

                audio_id += 1
                await asyncio.sleep(duration)

        await asyncio.gather(sender(), receiver())


# ================= STREAMLIT UI =================
st.set_page_config(page_title="Audio Streaming UI", layout="centered")
st.title("🎧 Audio Streaming (5s chunk + WS)")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3"]
)

chunk_sec = st.slider(
    "Chunk duration (seconds)",
    min_value=1,
    max_value=10,
    value=5
)

if uploaded_file:
    st.audio(uploaded_file.read())

    if st.button("🚀 Start Streaming"):
        log_box = st.empty()
        logs = []

        def log_cb(msg):
            logs.append(msg)
            log_box.text("\n".join(logs[-20:]))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.getvalue())
            wav_path = f.name

        try:
            asyncio.run(
                stream_audio(WS_URL, wav_path, chunk_sec, log_cb)
            )
        except RuntimeError:
            st.warning("⚠️ Async loop already running")

    if st.button("🚀 End Streaming"):
        st.success("✅ Streaming finished")