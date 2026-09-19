import json, urllib.request

SESSION_ID = "24e3551486a04b3ab30105461bb5ffd6"

payload = {
    "sequence_number": 1,
    "timestamp_start": 9.936,
    "timestamp_end": 10.0,
    "received_at": 10.0,
    "sample_rate": 16000,
    "audio": [0.0] * 1024,
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
