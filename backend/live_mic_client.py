
from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import numpy as np
import sounddevice as sd
import websockets


SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
BACKEND = "ws://127.0.0.1:8000"
RECORD_SECONDS = 30


async def main() -> None:
    import urllib.request

    request = urllib.request.Request(
        "http://127.0.0.1:8000/api/sessions",
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=5) as response:
        session = json.loads(response.read().decode())

    session_id = session["session_id"]
    url = f"{BACKEND}/api/ws/{session_id}"

    print(f"SESSION: {session_id}")
    print("Connecting to backend...")

    async with websockets.connect(url, max_size=20_000_000) as ws:
        connected = json.loads(await ws.recv())
        print("CONNECTED:", connected)

        sequence = 0
        start_time = time.monotonic()

        print("Speak into the microphone for up to 30 seconds.")
        print("The backend will process each completed 10-second window.")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:

            while time.monotonic() - start_time < RECORD_SECONDS:
                audio, overflowed = stream.read(CHUNK_SAMPLES)

                waveform = np.asarray(
                    audio[:, 0],
                    dtype=np.float32,
                )

                timestamp_start = sequence * CHUNK_SECONDS
                timestamp_end = timestamp_start + CHUNK_SECONDS

                payload = {
                    "audio": waveform.tolist(),
                    "sample_rate": SAMPLE_RATE,
                    "sequence_number": sequence,
                    "timestamp_start": timestamp_start,
                    "timestamp_end": timestamp_end,
                    "received_at": time.time(),
                    "packet_loss_before": 0,
                    "jitter_ms": None,
                }

                await ws.send(json.dumps(payload))
                response = json.loads(await ws.recv())

                if response.get("status") == "buffering":
                    print(
                        f"chunk={sequence} | buffering | "
                        f"window_ready={response.get('window_ready')}"
                    )
                elif response.get("status") == "error":
                    print("ERROR:", response)
                else:
                    print(
                        f"WINDOW {sequence}: "
                        f"prediction={response.get('prediction')} | "
                        f"spoof={response.get('spoof_probability', 0):.4f} | "
                        f"risk={response.get('risk', {}).get('score')} | "
                        f"level={response.get('risk', {}).get('level')} | "
                        f"temporal={response.get('temporal', {})}"
                    )

                if overflowed:
                    print("WARNING: microphone input overflow")

                sequence += 1

    print("LIVE MICROPHONE TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
