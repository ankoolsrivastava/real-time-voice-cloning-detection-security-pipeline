import json, wave, struct, urllib.request

SESSION_ID = "24e3551486a04b3ab30105461bb5ffd6"
AUDIO_PATH = r"D:\VoiceGaurd\dataset_v1\processed\real\hindi\REAL_000575.wav"

with wave.open(AUDIO_PATH, "rb") as wf:
    sr = wf.getframerate()
    frames = wf.readframes(sr * 10)

samples = struct.unpack("<" + "h" * (len(frames)//2), frames)
audio = [x / 32768.0 for x in samples]

payload = {
    "sequence_number": 0,
    "timestamp_start": 0.0,
    "timestamp_end": 10.0,
    "received_at": 10.0,
    "sample_rate": sr,
    "audio": audio,
    "packet_loss_before": 30,
    "jitter_ms": 80.0,
}

req = urllib.request.Request(
    f"http://127.0.0.1:8000/api/sessions/{SESSION_ID}/chunks",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as r:
    print(r.status)
    print(r.read().decode())

