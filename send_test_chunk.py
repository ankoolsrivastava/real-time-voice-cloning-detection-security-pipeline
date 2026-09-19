import json
import wave
import urllib.request

SESSION_ID = "33a12b641c1c43e2ab29a909ba8b58df"
AUDIO_PATH = r"D:\VoiceGaurd\dataset_v1\processed\real\hindi\REAL_000575.wav"
URL = f"http://127.0.0.1:8000/api/sessions/{SESSION_ID}/chunks"

with wave.open(AUDIO_PATH, "rb") as wf:
    sr = wf.getframerate()
    channels = wf.getnchannels()
    frames = wf.readframes(wf.getnframes())

print("sample_rate =", sr)
print("channels =", channels)
print("frames =", len(frames) // 2)

# PCM16 -> float32
import struct
samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
audio = [x / 32768.0 for x in samples]

payload = {
    "sequence_number": 0,
    "timestamp_start": 0.0,
    "timestamp_end": len(audio) / sr,
    "received_at": 0.0,
    "sample_rate": sr,
    "audio": audio,
    "packet_loss_before": 0.0,
    "jitter_ms": 0.0,
}

data = json.dumps(payload).encode("utf-8")

req = urllib.request.Request(
    URL,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as response:
    print(response.status)
    print(response.read().decode())
