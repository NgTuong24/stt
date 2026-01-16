import os
import time
import torch
import numpy as np
from typing import List, Dict
import azure.cognitiveservices.speech as speechsdk
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv(".env")
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
DEVICE = "cpu"              # cpu | cuda | mps
WORKERS = 4

# VAD
VAD_ONSET = 0.5
VAD_OFFSET = 0.363
VAD_CHUNK_SIZE = 10    # seconds
SAMPLE_RATE = 16000

TAIL_SECONDS = 0.5

LANGUAGE = "vi-VN" # en-US | vi-VN | ja-JP | ...
# LANGUAGE = "ja-JP" 
def get_device():
    if DEVICE == "cuda":
        return f"cuda"
    return DEVICE

def chunk_audio(
    audio: np.ndarray,
    sr: int,
    segments: List[Dict],
    tail_seconds: float = 0.5,
    end_stream: bool = False
):
    chunks = []
    tail_audio = None

    if not segments:
        return chunks, None
    
    tail_samples = int(tail_seconds * sr)

    if len(segments) > 1:
        for seg in segments[:-1]:
            start = int(seg["start"] * sr)
            end = int(seg["end"] * sr)
            chunks.append(audio[start:end])

    last_seg = segments[-1]
    seg_start = int(last_seg["start"] * sr)
    seg_end = int(last_seg["end"] * sr)

    if seg_end > seg_start:
        tail_start = max(0, seg_start - tail_samples)
        tail_audio = audio[tail_start:seg_end]
    
    if end_stream:
        chunks.append(tail_audio)
        tail_audio = None
    return chunks, tail_audio


class SileroVAD:
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        self.model.to(get_device())
        self.get_speech_timestamps = utils[0]

    def get_segments(self, audio: np.ndarray, sr: int):
        wav = torch.from_numpy(audio).to(get_device())
        segments = self.get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=sr,
            threshold=VAD_ONSET,
            min_silence_duration_ms=int(VAD_OFFSET * 1000),
        )

        return [
            {
                "start": s["start"] / sr,
                "end": s["end"] / sr
            }
            for s in segments
        ]


class AzureASR:
    def __init__(self, speech_key, speech_region, language="vi-VN"):
        self.speech_key = speech_key
        self.speech_region = speech_region
   
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        self.speech_config.speech_recognition_language = language

        self.audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=SAMPLE_RATE,
            bits_per_sample=16,
            channels=1
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """
        audio: numpy array float32 [-1, 1], mono, 16kHz
        """

        push_stream = speechsdk.audio.PushAudioInputStream(self.audio_format)
        audio_config = speechsdk.AudioConfig(stream=push_stream)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # float → PCM16
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

        push_stream.write(audio.tobytes())
        push_stream.close()

        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return ""
        else:
            return ""
        
    def multi_transcribe(self, audios: List[np.ndarray], max_workers=4) -> List[Dict]:
        results = [None] * len(audios)  

        def worker(idx, audio):
            text = self.transcribe(audio)
            return idx, text

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, idx, audio) for idx, audio in enumerate(audios)]
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text.strip()
        return results
    
    
class ASRPipeline:
    def __init__(self):
        self.stt_model = self.init_azure_stt()
        self.vad = SileroVAD()
        self.tail_audio = None

    def init_azure_stt(self):
        try:
            azure_asr = AzureASR(SPEECH_KEY, SPEECH_REGION, language=LANGUAGE)
            return azure_asr
        except Exception as e:
            print("Error initializing Azure ASR:", e)
            return None
        
    def __call__(self, audio: np.ndarray, end_stream: bool=False) -> Dict:
        time_str = time.time()
        if self.tail_audio is not None:
            audio = np.concatenate([self.tail_audio, audio])
            
        if self.vad:
            segments = self.vad.get_segments(audio, SAMPLE_RATE)
        else:
            segments = [{"start": 0, "end": len(audio) / SAMPLE_RATE}]
        
        
        chunks, self.tail_audio = chunk_audio(audio, SAMPLE_RATE, segments, end_stream=end_stream)
        
        results = []        
        time_str = time.time()  
        texts = self.stt_model.multi_transcribe(chunks, max_workers=WORKERS)
        for i, chunk in enumerate(chunks):
            if len(chunk) < SAMPLE_RATE:
                continue
            text = texts[i]
            results.append({
                "segment_id": i,
                "start": segments[i]["start"],
                "end": segments[i]["end"],
                "text": text.strip()
            })
        print("TIME transcribe with multi request:", time.time() - time_str)
        
        return {
            "segments": results,
            "full_text": " ".join([r["text"] for r in results])
        }


# if __name__ == "__main__":
#     audio_path = "quat0.75x.mp3"
#     pipeline = ASRPipeline()
    
#     audio = load_audio_by_file(audio_path)
#     result = pipeline(audio)
#     # print(result)
#     print("\n========== TRANSCRIPT ==========")
#     print(result["full_text"])

