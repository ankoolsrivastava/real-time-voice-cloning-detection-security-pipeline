import json
import urllib.request

SESSION_ID = "33a12b641c1c43e2ab29a909ba8b58df"
URL = f"http://127.0.0.1:8000/api/sessions/{SESSION_ID}/chunks"

audio = [0.0] * 1024

payload = {
    "sequence_number": 1,
    "timestamp_start": 9.936,
    "timestamp_end": 10.0,
    "received_at": 10.0,
    "sample_rate": 16000,
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
