import librosa
import numpy as np
SAMPLE_RATE=16000


def load_audio_by_file(path: str, sr=SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio