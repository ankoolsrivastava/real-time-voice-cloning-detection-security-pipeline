from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:\VoiceGaurd")

TXT_OUT = ROOT / "VOICEGUARD_PROJECT_AUDIT_V2.txt"
PDF_OUT = ROOT / "VOICEGUARD_PROJECT_AUDIT_V2.pdf"

# ============================================================
# DIRECTORIES WE NEVER NEED TO CRAWL
# ============================================================

EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "ml_env",
    "ai_env",
    "venv",
    "env",
    ".venv",
}

# These can contain thousands of audio/generated files.
# We record the directory itself but DO NOT enumerate contents.
HEAVY_DIR_NAMES = {
    "raw",
    "processed",
    "audio",
    "recordings",
    "samples",
    "generated",
    "outputs",
    "checkpoints",
    "cache",
    "caches",
    "temp",
    "tmp",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".go", ".rs"
}

DOC_EXTENSIONS = {
    ".txt", ".md", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf"
}

DATA_EXTENSIONS = {
    ".csv", ".tsv"
}

MODEL_EXTENSIONS = {
    ".pt", ".pth", ".pkl", ".pickle",
    ".onnx", ".safetensors", ".bin"
}

AUDIO_EXTENSIONS = {
    ".wav", ".m4a", ".mp3", ".flac",
    ".ogg", ".aac", ".opus"
}

IMPORTANT_KEYWORDS = [
    "backend",
    "frontend",
    "api",
    "fastapi",
    "inference",
    "predictor",
    "predict",
    "model_loader",
    "preprocess",
    "preprocessing",
    "postprocess",
    "postprocessing",
    "risk",
    "quality",
    "prosody",
    "evidence",
    "contract",
    "stream",
    "streaming",
    "buffer",
    "session",
    "websocket",
    "socket",
    "handoff",
    "integration",
    "test",
    "model",
    "checkpoint",
]

FROZEN_CHECKPOINT = (
    ROOT
    / "experiments"
    / "reverb_v2"
    / "models"
    / "best_model.pt"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b"
)


# ============================================================
# HELPERS
# ============================================================

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def size_text(path: Path) -> str:
    try:
        size = path.stat().st_size
    except Exception:
        return "UNKNOWN"

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def modified(path: Path) -> str:
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds")
    except Exception:
        return "UNKNOWN"


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception as exc:
        return f"HASH_ERROR: {exc}"


def read_text(path: Path, max_bytes: int = 3_000_000) -> str:
    try:
        data = path.read_bytes()

        if len(data) > max_bytes:
            data = data[:max_bytes]

        for encoding in (
            "utf-8-sig",
            "utf-8",
            "cp1252",
            "latin-1",
        ):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                pass

        return data.decode(
            "utf-8",
            errors="replace"
        )

    except Exception as exc:
        return f"[READ ERROR: {exc}]"


def keyword_matches(path: Path) -> list[str]:
    lower = rel(path).lower()

    return [
        k
        for k in IMPORTANT_KEYWORDS
        if k in lower
    ]


def classify(path: Path) -> str:
    ext = path.suffix.lower()

    if ext in CODE_EXTENSIONS:
        return "SOURCE"

    if ext in DOC_EXTENSIONS:
        return "DOCUMENT"

    if ext in DATA_EXTENSIONS:
        return "DATA"

    if ext in MODEL_EXTENSIONS:
        return "MODEL_ARTIFACT"

    if ext in AUDIO_EXTENSIONS:
        return "AUDIO"

    return "OTHER"


# ============================================================
# SAFE DIRECTORY WALK
# ============================================================

def collect_relevant_files():
    """
    Walk the project while pruning heavy directories BEFORE
    enumerating their files.
    """

    relevant = []
    heavy_dirs = []
    all_dirs = []

    for current_root, dirs, files in os.walk(ROOT):

        current = Path(current_root)

        # Prune virtual environments and caches.
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIR_NAMES
        ]

        # Record heavy directories but don't enter them.
        kept_dirs = []

        for dirname in dirs:
            child = current / dirname

            all_dirs.append(child)

            if dirname in HEAVY_DIR_NAMES:
                heavy_dirs.append(child)
            else:
                kept_dirs.append(dirname)

        dirs[:] = kept_dirs

        for filename in files:
            path = current / filename

            # Never include our generated report itself.
            if path in {TXT_OUT, PDF_OUT}:
                continue

            ext = path.suffix.lower()

            # Audio files are NOT useful individually for this audit.
            if ext in AUDIO_EXTENSIONS:
                continue

            # Model binaries are inventoried but never opened here.
            relevant.append(path)

    return relevant, heavy_dirs, all_dirs


# ============================================================
# DIRECTORY TREE
# ============================================================

def build_directory_tree():
    lines = []

    def walk(path: Path, prefix: str = ""):

        try:
            children = sorted(
                [
                    p
                    for p in path.iterdir()
                    if p.name not in EXCLUDED_DIR_NAMES
                    and p.name not in {
                        TXT_OUT.name,
                        PDF_OUT.name,
                    }
                ],
                key=lambda p: (
                    p.is_file(),
                    p.name.lower()
                )
            )
        except Exception as exc:
            lines.append(
                prefix + f"[ERROR: {exc}]"
            )
            return

        for index, child in enumerate(children):

            last = index == len(children) - 1

            connector = (
                "└── "
                if last
                else "├── "
            )

            if child.is_dir():

                if child.name in HEAVY_DIR_NAMES:
                    lines.append(
                        prefix
                        + connector
                        + child.name
                        + "/ [HEAVY DIRECTORY - CONTENTS NOT ENUMERATED]"
                    )
                    continue

                lines.append(
                    prefix
                    + connector
                    + child.name
                    + "/"
                )

                next_prefix = (
                    prefix + ("    " if last else "│   ")
                )

                walk(
                    child,
                    next_prefix
                )

            else:

                lines.append(
                    prefix
                    + connector
                    + child.name
                    + f" [{size_text(child)}]"
                )

    lines.append("VoiceGuard/")

    walk(ROOT)

    return lines


# ============================================================
# PYTHON SOURCE ANALYSIS
# ============================================================

def analyze_python(path: Path) -> str:

    text = read_text(
        path,
        max_bytes=2_000_000
    )

    lines = text.splitlines()

    imports = []
    definitions = []

    for line in lines:

        stripped = line.strip()

        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
        ):
            imports.append(stripped)

        if (
            stripped.startswith("def ")
            or stripped.startswith("async def ")
            or stripped.startswith("class ")
        ):
            definitions.append(stripped)

    out = []

    out.append(
        f"Lines: {len(lines)}"
    )

    if imports:

        out.append("Imports:")

        for item in imports[:80]:
            out.append(
                "  " + item
            )

    if definitions:

        out.append("Definitions:")

        for item in definitions[:150]:
            out.append(
                "  " + item
            )

    # Search specifically for integration-relevant interfaces.
    interesting_terms = [
        "predict(",
        "FastAPI",
        "WebSocket",
        "websocket",
        "session",
        "audio_window",
        "spoof_probability",
        "risk_score",
        "prosody",
        "quality",
        "inference",
        "mock",
    ]

    found = []

    lower = text.lower()

    for term in interesting_terms:

        if term.lower() in lower:
            found.append(term)

    if found:

        out.append(
            "Integration-related tokens found:"
        )

        for term in found:
            out.append(
                "  - " + term
            )

    return "\n".join(out)


# ============================================================
# DOCUMENT CONTENT
# ============================================================

def document_report(files: list[Path]) -> str:

    docs = [
        p
        for p in files
        if p.suffix.lower() in DOC_EXTENSIONS
    ]

    docs.sort(
        key=lambda p: rel(p).lower()
    )

    output = []

    for path in docs:

        output.append("")
        output.append(
            "=" * 90
        )
        output.append(
            f"DOCUMENT: {rel(path)}"
        )
        output.append(
            "=" * 90
        )

        size = path.stat().st_size

        output.append(
            f"SIZE: {size_text(path)}"
        )

        output.append(
            f"MODIFIED: {modified(path)}"
        )

        if size > 3_000_000:

            output.append(
                "[CONTENT OMITTED: document exceeds 3 MB]"
            )

            continue

        output.append(
            read_text(path)
        )

    return "\n".join(output)


# ============================================================
# DATA FILE PREVIEW
# ============================================================

def data_report(files: list[Path]) -> str:

    data_files = [
        p
        for p in files
        if p.suffix.lower() in DATA_EXTENSIONS
    ]

    data_files.sort(
        key=lambda p: rel(p).lower()
    )

    output = []

    for path in data_files:

        output.append("")
        output.append(
            "-" * 80
        )
        output.append(
            f"DATA FILE: {rel(path)}"
        )
        output.append(
            "-" * 80
        )

        output.append(
            f"SIZE: {size_text(path)}"
        )

        text = read_text(
            path,
            max_bytes=30_000
        )

        lines = text.splitlines()

        output.append(
            "PREVIEW:"
        )

        for line in lines[:8]:
            output.append(line)

        if len(lines) > 8:
            output.append(
                f"... {len(lines) - 8} additional lines omitted"
            )

    return "\n".join(output)


# ============================================================
# MODEL ARTIFACT REPORT
# ============================================================

def model_report(files: list[Path]) -> str:

    models = [
        p
        for p in files
        if p.suffix.lower() in MODEL_EXTENSIONS
    ]

    # Explicitly include frozen checkpoint even if it sits somewhere
    # we did not crawl.
    if FROZEN_CHECKPOINT.exists():
        if FROZEN_CHECKPOINT not in models:
            models.append(FROZEN_CHECKPOINT)

    models.sort(
        key=lambda p: rel(p).lower()
    )

    output = []

    for path in models:

        output.append("")
        output.append(
            "-" * 80
        )
        output.append(
            f"MODEL ARTIFACT: {rel(path)}"
        )
        output.append(
            "-" * 80
        )

        output.append(
            f"SIZE: {size_text(path)}"
        )

        output.append(
            f"MODIFIED: {modified(path)}"
        )

        # Hash model artifacts only.
        output.append(
            f"SHA256: {sha256(path)}"
        )

        if path == FROZEN_CHECKPOINT:

            actual = sha256(path)

            if actual == EXPECTED_CHECKPOINT_SHA256:

                output.append(
                    "FROZEN V2 CHECKPOINT STATUS: MATCH"
                )

            else:

                output.append(
                    "FROZEN V2 CHECKPOINT STATUS: !!! HASH MISMATCH !!!"
                )

    return "\n".join(output)


# ============================================================
# IMPORTANT FILE REPORT
# ============================================================

def important_file_report(files: list[Path]) -> str:

    important = []

    for path in files:

        matches = keyword_matches(path)

        if matches:

            important.append(
                (
                    rel(path),
                    matches
                )
            )

    important.sort(
        key=lambda x: x[0].lower()
    )

    output = []

    for path, matches in important:

        output.append(
            f"{path}"
        )

        output.append(
            "  matches: "
            + ", ".join(matches)
        )

    return "\n".join(output)


# ============================================================
# KNOWN ML BASELINE
# ============================================================

def write_known_baseline(f):

    f.write(
        """
AUTHORITATIVE VOICEGUARD ML BASELINE
====================================

Model:
    VoiceGuard V2 epoch 8

Model status:
    FROZEN

Do NOT call this V3.

Checkpoint:
    D:\\VoiceGaurd\\experiments\\reverb_v2\\models\\best_model.pt

Expected frozen checkpoint SHA256:
    ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b

Operational threshold:
    0.25

Architecture:
    CNN + BiGRU + temporal attention

Parameters:
    1,143,331

Active ML/evidence path:
    Audio
      -> exact V2 preprocessing
      -> Log-Mel
      -> frozen V2
      -> spoof probability
      -> deployable prosody scorer
      -> audio quality/confidence
      -> active evidence fusion
      -> Dynamic Risk Engine V1
      -> operational risk score
      -> risk level

Current prosody fusion:
    V2 primary = 0.83
    Prosody = 0.17

Current languages validated:
    Hindi
    Marathi

Indian English:
    PENDING

Unseen cloning:
    PENDING

Speaker consistency:
    REJECTED / RESERVED

Temporal accumulator:
    IMPLEMENTED + UNIT TESTED
    Live streaming integration = PENDING

Live microphone/telephony:
    SYSTEM/BACKEND INTEGRATION PENDING

Risk score:
    0-100 operational score
    NOT a calibrated probability

Audio quality:
    Confidence mechanism
    Poor quality must NOT become spoof evidence.

"""
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not ROOT.exists():

        raise SystemExit(
            f"ERROR: {ROOT} does not exist."
        )

    print(
        "VoiceGuard Audit V2 starting..."
    )

    print(
        f"Project root: {ROOT}"
    )

    files, heavy_dirs, all_dirs = (
        collect_relevant_files()
    )

    print(
        f"Relevant non-audio files: {len(files)}"
    )

    print(
        f"Heavy directories skipped: {len(heavy_dirs)}"
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    with TXT_OUT.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:

        f.write(
            "VOICEGUARD PROJECT AUDIT V2\n"
        )

        f.write(
            "=" * 90 + "\n"
        )

        f.write(
            f"Generated: "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
        )

        f.write(
            f"Project root: {ROOT}\n"
        )

        f.write(
            "\nThis is a READ-ONLY audit report.\n"
        )

        f.write(
            "The audit script does not modify project code, datasets, "
            "models, or configuration.\n"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "AUDIT SUMMARY\n"
            + "=" * 90
            + "\n"
        )

        f.write(
            f"Relevant files inspected: {len(files)}\n"
        )

        f.write(
            f"Heavy directories skipped: {len(heavy_dirs)}\n"
        )

        f.write(
            f"Directory records discovered: {len(all_dirs)}\n"
        )

        f.write(
            "\nHeavy directories were intentionally not enumerated "
            "because they may contain tens of thousands of audio/generated files.\n"
        )

        # ----------------------------------------------------
        # TREE
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "PROJECT DIRECTORY TREE\n"
            + "=" * 90
            + "\n"
        )

        for line in build_directory_tree():
            f.write(
                line + "\n"
            )

        # ----------------------------------------------------
        # IMPORTANT FILES
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "POTENTIALLY RELEVANT INTEGRATION FILES\n"
            + "=" * 90
            + "\n"
        )

        important = important_file_report(
            files
        )

        if important:
            f.write(
                important
            )
        else:
            f.write(
                "No keyword-matched files found.\n"
            )

        # ----------------------------------------------------
        # ALL NON-BINARY INVENTORY
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "NON-AUDIO FILE INVENTORY\n"
            + "=" * 90
            + "\n"
        )

        for path in sorted(
            files,
            key=lambda p: rel(p).lower()
        ):

            f.write(
                f"PATH: {rel(path)}\n"
            )

            f.write(
                f"TYPE: {classify(path)}\n"
            )

            f.write(
                f"SIZE: {size_text(path)}\n"
            )

            f.write(
                f"MODIFIED: {modified(path)}\n"
            )

            f.write("\n")

        # ----------------------------------------------------
        # MODEL ARTIFACTS
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "MODEL ARTIFACTS / CHECKPOINT HASHES\n"
            + "=" * 90
            + "\n"
        )

        f.write(
            model_report(files)
        )

        # ----------------------------------------------------
        # PYTHON FILES
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "PYTHON SOURCE ANALYSIS\n"
            + "=" * 90
            + "\n"
        )

        python_files = [
            p
            for p in files
            if p.suffix.lower() == ".py"
        ]

        python_files.sort(
            key=lambda p: rel(p).lower()
        )

        for path in python_files:

            f.write(
                "\n"
                + "-" * 80
                + "\n"
            )

            f.write(
                f"FILE: {rel(path)}\n"
            )

            f.write(
                "-" * 80
                + "\n"
            )

            f.write(
                analyze_python(path)
            )

            f.write("\n")

        # ----------------------------------------------------
        # DATA PREVIEWS
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "CSV / DATA FILE PREVIEWS\n"
            + "=" * 90
            + "\n"
        )

        f.write(
            data_report(files)
        )

        # ----------------------------------------------------
        # FULL DOCUMENTATION
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "FULL DOCUMENTATION / CONFIG CONTENT\n"
            + "=" * 90
            + "\n"
        )

        f.write(
            document_report(files)
        )

        # ----------------------------------------------------
        # AUTHORITATIVE BASELINE
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "AUTHORITATIVE PROJECT BASELINE\n"
            + "=" * 90
            + "\n"
        )

        write_known_baseline(f)

        # ----------------------------------------------------
        # BACKEND INTEGRATION QUESTIONS
        # ----------------------------------------------------

        f.write(
            "\n"
            + "=" * 90
            + "\n"
            + "BACKEND INTEGRATION AUDIT CHECKLIST\n"
            + "=" * 90
            + "\n"
        )

        questions = [
            "Where is the current mock inference implementation?",
            "Where is the backend inference interface?",
            "Does predictor.predict(audio_window) already exist?",
            "What exact type does audio_window use?",
            "Where is resampling performed?",
            "Where is mono conversion performed?",
            "Where is peak normalization performed?",
            "Where is 10-second pad/truncate performed?",
            "Where is Log-Mel extraction performed?",
            "Are the exact V2 feature parameters preserved?",
            "Is threshold 0.25 used?",
            "Is the frozen V2 checkpoint used?",
            "Is checkpoint SHA256 verified?",
            "Is prosody scorer integrated?",
            "Are fusion weights 0.83 / 0.17?",
            "Is audio quality used as confidence only?",
            "Is Dynamic Risk Engine V1 integrated?",
            "Is temporal accumulation connected to sessions?",
            "Does backend expose evidence_confidence?",
            "Does backend expose risk_score?",
            "Does backend expose risk_level?",
            "Does backend expose quality status?",
            "Does backend expose prosody evidence?",
            "Does frontend consume the evidence fields?",
            "Does backend keep ML model state session-independent?",
            "Does backend support concurrent inference sessions?",
            "Is streaming end-to-end tested?",
            "Are Indian English claims avoided?",
            "Are unseen cloning claims avoided?",
            "Is speaker consistency inactive?",
            "Are locked datasets untouched?",
            "Is V2 checkpoint unchanged?",
        ]

        for i, question in enumerate(
            questions,
            start=1
        ):

            f.write(
                f"{i:02d}. {question}\n"
            )

        f.write(
            "\nEND OF AUDIT\n"
        )

    print(
        f"\nTXT report created:\n{TXT_OUT}"
    )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Preformatted,
            Spacer,
        )
        from reportlab.lib.units import mm

        print(
            "Generating PDF..."
        )

        text = TXT_OUT.read_text(
            encoding="utf-8",
            errors="replace"
        )

        pdf = SimpleDocTemplate(
            str(PDF_OUT),
            pagesize=A4,
            rightMargin=10 * mm,
            leftMargin=10 * mm,
            topMargin=10 * mm,
            bottomMargin=10 * mm,
            title="VoiceGuard Project Audit V2",
        )

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "VoiceGuard Project Audit V2",
                styles["Title"]
            )
        )

        story.append(
            Spacer(1, 8)
        )

        lines = text.splitlines()

        block = []

        for line in lines:

            block.append(line)

            if len(block) >= 220:

                story.append(
                    Preformatted(
                        "\n".join(block),
                        styles["Code"],
                        maxLineLength=120,
                    )
                )

                story.append(
                    Spacer(1, 4)
                )

                block = []

        if block:

            story.append(
                Preformatted(
                    "\n".join(block),
                    styles["Code"],
                    maxLineLength=120,
                )
            )

        pdf.build(story)

        print(
            f"PDF report created:\n{PDF_OUT}"
        )

    except ImportError:

        print(
            "\nPDF skipped because reportlab is not installed."
        )

        print(
            "The TXT report is complete."
        )

    except Exception as exc:

        print(
            f"\nPDF generation failed: {exc}"
        )

        print(
            "The TXT report is still complete."
        )


if __name__ == "__main__":
    main()