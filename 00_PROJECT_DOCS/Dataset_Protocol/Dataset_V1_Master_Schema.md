# VoiceGuard Dataset V1 — Master Metadata Schema

## Purpose

This document defines the common metadata structure used by all
VoiceGuard Dataset V1 data pillars:

1. REAL
2. SPOOF
3. ROBUSTNESS

All three contributors must maintain compatible metadata so that their
datasets can be validated and merged into a single master manifest.

---

# Primary Classification Labels

The `label` field has only two valid values:

- `bonafide`
- `spoof`

Conditions such as noise, replay, reverb, compression, classroom,
phone recording, etc. are NOT classification labels.

Examples:

REAL + noise → bonafide

SPOOF + noise → spoof

This prevents the model from learning shortcuts such as:

"noisy audio = spoof".

---

# Master Columns

| Column            | Required | Description                                 |
| ----------------- | -------- | ------------------------------------------- |
| file_id           | YES      | Globally unique ID for the exact audio file |
| parent_file_id    | YES\*    | ID of source audio if this file is derived  |
| speaker_id        | YES\*    | Speaker/source identity where applicable    |
| language          | YES      | Language code                               |
| language_name     | YES      | Human-readable language                     |
| split             | YES      | train / validation / test                   |
| label             | YES      | bonafide / spoof                            |
| source            | YES      | Immediate audio source                      |
| source_dataset    | YES      | Dataset or collection name                  |
| original_filename | YES      | Original filename                           |
| source_path       | YES      | Current project path                        |
| generator         | YES      | Generation method; `none` for natural audio |
| generator_version | YES      | Generator/model version if applicable       |
| condition         | YES      | Main audio condition                        |
| environment       | NO       | Recording/environment condition             |
| device            | NO       | Recording/playback device                   |
| distance_m        | NO       | Approximate speaker/device distance         |
| transcript        | NO       | Spoken/generated text                       |
| text_id           | NO       | ID of source text/prompt                    |
| duration_sec      | YES      | Audio duration in seconds                   |
| sample_rate       | YES      | Audio sampling rate                         |
| channels          | YES      | Number of audio channels                    |
| quality_status    | YES      | QC status                                   |
| notes             | NO       | Additional information                      |

`*` may be `NA` when the field genuinely does not apply.

---

# File ID Rules

Every audio file must have exactly one unique `file_id`.

Example:

REAL_HI_000001

SPOOF_HI_000001

ROBUST_HI_000001

Duplicate file IDs are not allowed.

---

# Parent File Rules

`parent_file_id` identifies the audio from which a derived sample was
created.

Original audio:

parent_file_id = NA

Derived audio:

parent_file_id = original_file_id

Example:

REAL_HI_000001
↓
ROBUST_HI_000001

ROBUST_HI_000001:

parent_file_id = REAL_HI_000001

Parent lineage must be preserved.

---

# Label Rules

Allowed values:

bonafide
spoof

Examples:

Natural Hindi speech:
label = bonafide

TTS Hindi speech:
label = spoof

Cloned Hindi speech:
label = spoof

Bonafide speech with classroom noise:
label = bonafide

Spoof speech with classroom noise:
label = spoof

---

# Language Scope

Current Dataset V1 languages:

HI = Hindi
MR = Marathi

Indian English is currently deferred and is not part of the core
Dataset V1 scope.

---

# Split Rules

Allowed values:

train
validation
test

Splits must prevent speaker leakage.

A speaker must not appear across multiple splits.

Derived audio must preserve the split of its parent unless the master
dataset procedure explicitly determines otherwise.

Contributors must NOT independently create conflicting train,
validation and test assignments.

The master split policy is controlled by the project lead.

---

# Generator Rules

Natural/real audio:

generator = none
generator_version = none

Synthetic speech may use:

generator = TTS
generator = voice_cloning
generator = voice_conversion

The exact generator/model/version must be recorded where available.

---

# Condition Rules

`condition` describes how the audio exists or what transformation/
attack condition applies.

Examples:

natural
synthetic
noise
replay
reverb
compression
device_recording

Condition must not replace the `label`.

---

# Missing Metadata

Do not invent metadata.

If information genuinely does not exist:

NA

may be used.

For example, if a source dataset does not provide age or device
information, do not guess it.

---

# Audio Standard

The current REAL dataset has been normalized/verified as:

16 kHz
mono
16-bit PCM WAV

Future data should follow the same standard where technically
appropriate.

---

# Quality Status

Quality status describes the QC result.

Current REAL dataset values include:

usable
excluded_duration

Future validation may additionally use:

excluded
invalid

Only files marked `usable` should enter the final training dataset.

---

# Three-Pillar Dataset Design

## REAL

Natural human speech.

label = bonafide

Examples:

Hindi
Marathi
natural recording

---

## SPOOF

Synthetic or manipulated speech.

label = spoof

Examples:

TTS
voice cloning
voice conversion

---

## ROBUSTNESS

Modified/recorded audio used to test environmental and channel
robustness.

The original classification label is preserved.

Examples:

REAL + noise → bonafide

REAL + reverb → bonafide

SPOOF + noise → spoof

SPOOF + replay → spoof

---

# Metadata Merge Rule

Each contributor may maintain an individual manifest:

real_manifest.csv
spoof_manifest.csv
robustness_manifest.csv

All manifests must follow this schema.

They will eventually be merged into:

dataset_v1_manifest.csv

The master manifest must pass:

- required-column validation
- duplicate ID validation
- duplicate path validation
- language validation
- label validation
- split validation
- duration validation
- speaker leakage validation
- parent lineage validation
- audio existence validation
- metadata completeness checks

before Dataset V1 is frozen.

---

# Important Principle

Dataset V1 must be reproducible and traceable.

Every sample should answer:

1. What is this file?
2. Who/source generated it?
3. What language is it?
4. Is it bonafide or spoof?
5. What condition was applied?
6. What was its parent file?
7. Which split does it belong to?
8. Has it passed quality control?

No fabricated metadata should be introduced.
