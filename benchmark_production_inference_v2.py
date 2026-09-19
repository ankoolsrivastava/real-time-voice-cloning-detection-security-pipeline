from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(r"D:\VoiceGaurd")
SCRIPTS = ROOT / "scripts"
EXPERIMENT = ROOT / "experiments" / "reverb_v2"
CHECKPOINT = EXPERIMENT / "models" / "best_model.pt"
MANIFEST = ROOT / "master_v1" / "metadata" / "master_dataset_v1.csv"
OUT_DIR = EXPERIMENT / "production_inference_benchmark"
OUT_JSON = OUT_DIR / "V2_PRODUCTION_INFERENCE_BENCHMARK.json"
OUT_CSV = OUT_DIR / "V2_PRODUCTION_INFERENCE_LATENCIES.csv"

import sys
sys.path.insert(0, str(SCRIPTS))

from audio_preprocessor import VoiceGuardAudioPreprocessor
from feature_extractor import VoiceGuardFeatureExtractor
from model import VoiceGuardModel


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_audio_column(fieldnames):
    preferred = [
        "audio_path", "filepath", "file_path", "path",
        "processed_path", "audio_file", "file"
    ]
    lowered = {x.lower(): x for x in fieldnames}
    for p in preferred:
        if p in lowered:
            return lowered[p]
    for x in fieldnames:
        lx = x.lower()
        if "audio" in lx and "path" in lx:
            return x
    raise RuntimeError(f"No audio-path column found. Columns: {fieldnames}")


def pick_samples():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    rows = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        audio_col = find_audio_column(reader.fieldnames or [])
        for row in reader:
            p = Path(row[audio_col])
            if p.exists() and p.is_file():
                rows.append((p, row))

    # Deterministic, small benchmark set. Never modifies the manifest/audio.
    selected = []
    seen = set()
    targets = [
        ("bonafide", "hindi"),
        ("bonafide", "marathi"),
        ("spoof", "hindi"),
        ("spoof", "marathi"),
    ]

    def norm(v):
        return str(v or "").strip().lower()

    for label, lang in targets:
        for p, row in rows:
            key = (norm(row.get("label")), norm(row.get("language")))
            if key == (label, lang) and str(p) not in seen:
                selected.append((p, row))
                seen.add(str(p))
                break

    # Add a few more deterministic files for stable timing statistics.
    for p, row in rows:
        if len(selected) >= 12:
            break
        if str(p) not in seen:
            selected.append((p, row))
            seen.add(str(p))

    if len(selected) < 4:
        raise RuntimeError(f"Only {len(selected)} usable benchmark files found.")

    return selected[:12]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("VOICEGUARD V2 — PRODUCTION INFERENCE BENCHMARK")
    print("=" * 72)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"V2 checkpoint missing: {CHECKPOINT}")
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Master manifest missing: {MANIFEST}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device     :", device)
    if device.type == "cuda":
        print("GPU        :", torch.cuda.get_device_name(0))
    print("Checkpoint :", CHECKPOINT)
    print("SHA256     :", sha256(CHECKPOINT))

    pre = VoiceGuardAudioPreprocessor(target_sr=16000, max_duration_sec=10.0)
    feat = VoiceGuardFeatureExtractor(sample_rate=16000)

    model = VoiceGuardModel().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint missing model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Validation loss :", checkpoint.get("validation_loss"))
    print("Validation acc  :", checkpoint.get("validation_accuracy"))

    samples = pick_samples()
    print("Benchmark files :", len(samples))

    # Warm-up on a real audio sample, not synthetic features.
    warm_audio = pre.process(str(samples[0][0]))
    warm_x = feat.extract_with_channel(warm_audio).unsqueeze(0).to(device)
    for _ in range(10):
        with torch.inference_mode():
            model(warm_x)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records = []

    for idx, (path, row) in enumerate(samples, 1):
        # Time the complete production path:
        # disk read -> preprocessing -> Log-Mel -> model inference.
        t0 = time.perf_counter()
        waveform = pre.process(str(path))
        t1 = time.perf_counter()
        x = feat.extract_with_channel(waveform).unsqueeze(0).to(device)
        sync(device)
        t2 = time.perf_counter()
        with torch.inference_mode():
            logits, attention = model(x)
            probs = torch.softmax(logits, dim=1)
        sync(device)
        t3 = time.perf_counter()

        spoof_prob = float(probs[0, 1].item())

        records.append({
            "index": idx,
            "audio_path": str(path),
            "label": row.get("label", ""),
            "language": row.get("language", ""),
            "condition": row.get("condition", ""),
            "generator": row.get("generator", ""),
            "preprocess_ms": (t1 - t0) * 1000,
            "feature_ms": (t2 - t1) * 1000,
            "model_ms": (t3 - t2) * 1000,
            "end_to_end_ms": (t3 - t0) * 1000,
            "spoof_probability": spoof_prob,
            "feature_shape": list(x.shape),
            "prediction": "SPOOF" if spoof_prob >= 0.25 else "BONAFIDE",
        })

        print(
            f"[{idx:02d}/{len(samples)}] "
            f"pre={records[-1]['preprocess_ms']:.2f}ms "
            f"feat={records[-1]['feature_ms']:.2f}ms "
            f"model={records[-1]['model_ms']:.2f}ms "
            f"e2e={records[-1]['end_to_end_ms']:.2f}ms"
        )

    # Repeat model-only timing on one real feature tensor to estimate steady-state
    # neural-network latency independently of disk/audio preprocessing.
    x = warm_x
    model_times = []
    for _ in range(100):
        sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(x)
        sync(device)
        model_times.append((time.perf_counter() - t0) * 1000)

    # Throughput is measured for the complete production path over the selected
    # real files, not a fabricated tensor benchmark.
    total_e2e_sec = sum(r["end_to_end_ms"] for r in records) / 1000.0
    throughput = len(records) / total_e2e_sec if total_e2e_sec > 0 else None

    result = {
        "benchmark_version": "voiceguard_v2_production_inference_benchmark_v1",
        "model_version": "voiceguard_v2_epoch8",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "checkpoint_validation_accuracy": checkpoint.get("validation_accuracy"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "preprocessing": {
            "sample_rate": 16000,
            "mono": True,
            "max_duration_sec": 10.0,
            "peak_normalization": True,
        },
        "feature_extraction": {
            "type": "log_mel",
            "n_fft": 400,
            "hop_length": 160,
            "win_length": 400,
            "n_mels": 64,
            "fmin": 20,
            "fmax": 8000,
        },
        "threshold": 0.25,
        "sample_count": len(records),
        "complete_pipeline": {
            "mean_ms": statistics.mean(r["end_to_end_ms"] for r in records),
            "median_ms": statistics.median(r["end_to_end_ms"] for r in records),
            "min_ms": min(r["end_to_end_ms"] for r in records),
            "max_ms": max(r["end_to_end_ms"] for r in records),
            "throughput_files_per_sec": throughput,
        },
        "steady_state_model_only": {
            "runs": len(model_times),
            "mean_ms": statistics.mean(model_times),
            "median_ms": statistics.median(model_times),
            "p95_ms": sorted(model_times)[94],
            "min_ms": min(model_times),
            "max_ms": max(model_times),
        },
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "notes": [
            "Benchmark uses the frozen V2 epoch-8 checkpoint.",
            "Benchmark does not modify locked datasets or the checkpoint.",
            "Timing includes real-file preprocessing, feature extraction, and model inference.",
            "Model-only timing is reported separately.",
            "Results are hardware/runtime measurements, not deployment SLA claims.",
        ],
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"JSON : {OUT_JSON}")
    print(f"CSV  : {OUT_CSV}")
    print(f"E2E mean       : {result['complete_pipeline']['mean_ms']:.2f} ms")
    print(f"E2E median     : {result['complete_pipeline']['median_ms']:.2f} ms")
    print(f"E2E throughput  : {throughput:.2f} files/s")
    print(f"Model median   : {result['steady_state_model_only']['median_ms']:.2f} ms")
    print(f"Peak GPU memory: {result['gpu_peak_memory_mb']:.2f} MB" if result["gpu_peak_memory_mb"] is not None else "Peak GPU memory: N/A")
    print("=" * 72)


if __name__ == "__main__":
    main()
