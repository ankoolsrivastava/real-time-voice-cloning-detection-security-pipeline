from pathlib import Path
from datetime import datetime
import os
import sys

ROOT = Path(r"D:\VoiceGaurd")
OUT = ROOT / "project_structure_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Keep the audit useful without traversing giant environments/caches.
EXCLUDED_DIRS = {
    ".git",
    "ml_env",
    "hf_cache",
    "__pycache__",
    ".pytest_cache",
}

def should_skip(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)

entries = []

# Include directories and files, sorted naturally.
for current, dirs, files in os.walk(ROOT):
    current_path = Path(current)

    # Prune excluded directories before os.walk descends into them.
    dirs[:] = sorted([d for d in dirs if d not in EXCLUDED_DIRS], key=str.lower)
    files = sorted(files, key=str.lower)

    if should_skip(current_path):
        continue

    rel = current_path.relative_to(ROOT)
    depth = 0 if str(rel) == "." else len(rel.parts)

    if str(rel) != ".":
        entries.append(("dir", depth, current_path.name, current_path))

    for name in files:
        p = current_path / name
        if should_skip(p):
            continue
        entries.append(("file", depth + 1, name, p))

# Build readable tree.
tree = [f"D:\\VoiceGaurd\\"]

# Re-scan in a deterministic tree order.
def add_tree(path: Path, prefix=""):
    children = []
    try:
        children = [p for p in path.iterdir() if p.name not in EXCLUDED_DIRS]
    except PermissionError:
        tree.append(prefix + "[ACCESS DENIED]")
        return

    children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

    for i, p in enumerate(children):
        last = i == len(children) - 1
        branch = "└── " if last else "├── "
        suffix = "/" if p.is_dir() else ""
        tree.append(prefix + branch + p.name + suffix)

        if p.is_dir():
            add_tree(p, prefix + ("    " if last else "│   "))

add_tree(ROOT)

top_dirs = sorted(
    [p for p in ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDED_DIRS],
    key=lambda p: p.name.lower()
)

file_count = sum(1 for kind, *_ in entries if kind == "file")
dir_count = sum(1 for kind, *_ in entries if kind == "dir")

report = []
report += [
    "VOICEGUARD PROJECT STRUCTURE AUDIT",
    "=" * 36,
    "",
    f"Root: {ROOT}",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "PURPOSE",
    "-------",
    "Read-only inventory of the VoiceGuard project structure.",
    "This script does not rename, delete, copy, modify, stage, or commit project files.",
    "",
    "EXCLUDED DIRECTORIES",
    "--------------------",
    ", ".join(sorted(EXCLUDED_DIRS)),
    "",
    "INVENTORY",
    "---------",
    f"Top-level directories: {len(top_dirs)}",
    f"Directories inventoried: {dir_count}",
    f"Files inventoried: {file_count}",
    "",
    "TOP-LEVEL DIRECTORIES",
    "---------------------",
]

for p in top_dirs:
    report.append(f"- {p.name}")

report += [
    "",
    "FULL PROJECT TREE",
    "-----------------",
    *tree,
    "",
    "AUDIT NOTES",
    "-----------",
    "1. This is a structural inventory, not a code-quality audit.",
    "2. Large environment/cache directories are excluded to keep the report manageable.",
    "3. Locked datasets and experiment artifacts are only listed; they are not changed.",
    "4. This report should be reviewed before creating the final handoff structure.",
]

txt_path = OUT / "VoiceGuard_Project_Structure_Audit.txt"
txt_path.write_text("\n".join(report), encoding="utf-8")

# PDF generation using reportlab.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.units import mm
except ImportError:
    print("ERROR: reportlab is not installed in the active Python environment.")
    print("Run: python -m pip install reportlab")
    sys.exit(2)

pdf_path = OUT / "VoiceGuard_Project_Structure_Audit.pdf"

doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=A4,
    rightMargin=12 * mm,
    leftMargin=12 * mm,
    topMargin=12 * mm,
    bottomMargin=12 * mm,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "AuditTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    alignment=TA_LEFT,
    spaceAfter=8,
)
mono_style = ParagraphStyle(
    "AuditMono",
    fontName="Courier",
    fontSize=6.7,
    leading=8.2,
    spaceAfter=0,
)

story = [
    Paragraph("VoiceGuard Project Structure Audit", title_style),
    Paragraph(
        f"Root: D:\\VoiceGaurd<br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"],
    ),
    Spacer(1, 6),
    Preformatted("\n".join(report), mono_style),
]

doc.build(story)

print()
print("=" * 50)
print("VOICEGUARD STRUCTURE AUDIT COMPLETE")
print("=" * 50)
print(f"PDF : {pdf_path}")
print(f"TXT : {txt_path}")
print(f"Top-level directories: {len(top_dirs)}")
print(f"Directories inventoried: {dir_count}")
print(f"Files inventoried: {file_count}")
print()
