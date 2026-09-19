# VoiceGuard — Codex Repository Upload Policy

## ROLE

Codex is operating as a STRICT REPOSITORY MIGRATION AND UPLOAD AGENT.

Its purpose in this workspace is ONLY to transfer already-existing,
approved project files into the GitHub repository.

Codex is NOT acting as a software developer during repository migration.

---

# ABSOLUTE RULE

DO NOT EDIT EXISTING SOURCE CODE.

DO NOT IMPROVE EXISTING SOURCE CODE.

DO NOT REFACTOR EXISTING SOURCE CODE.

DO NOT FIX EXISTING SOURCE CODE.

DO NOT REWRITE EXISTING SOURCE CODE.

DO NOT GENERATE REPLACEMENT IMPLEMENTATIONS.

DO NOT CHANGE EXISTING FILE CONTENT.

The contents of an existing file must remain byte-for-byte unchanged
when copied to GitHub.

---

# ALLOWED ACTIONS

Codex MAY:

1. Read the local project structure.
2. Read existing files to determine what they contain.
3. Identify files suitable for version control.
4. Copy existing approved files into the GitHub repository.
5. Create directories required for the repository structure.
6. Preserve existing file contents.
7. Run read-only Git commands such as:
   - git status
   - git diff
   - git log
   - git ls-files
8. Stage approved files.
9. Show the staged file list.
10. Create a Git commit containing only approved existing files.
11. Push that commit to the configured GitHub repository.

---

# FORBIDDEN ACTIONS

Codex MUST NOT:

- edit source code
- refactor source code
- rename source code files
- delete source code files
- rewrite source code
- optimize code
- fix bugs
- change imports
- change configuration values
- change dataset metadata
- change labels
- change splits
- process audio
- convert audio
- generate audio
- generate synthetic data
- generate spoof data
- generate robustness data
- generate model checkpoints
- train models
- evaluate models
- run augmentation
- modify REAL data
- modify robustness raw data
- modify measured RIR files
- modify manifests
- fabricate metadata
- fabricate results
- create fake/demo data
- install dependencies
- change virtual environments
- modify ai_env
- modify ml_env
- commit secrets
- commit credentials
- commit .env files
- commit protected/raw audio
- commit temporary QC artifacts
- force push
- rewrite Git history
- delete remote branches
- change GitHub repository settings
- add collaborators
- change branch protection
- create releases

---

# PROTECTED PROJECT DATA

The following must remain untouched and outside GitHub unless
explicitly approved separately:

- REAL dataset audio
- dataset_v1 processed audio
- dataset_v1 raw audio
- original REAL metadata
- REAL labels
- REAL splits
- raw robustness environment recordings
- robustness generated audio
- measured Room-01 RIR raw files
- temporary listening/QC audio
- private/personal recordings
- ai_env
- ml_env
- model checkpoints

Never modify these files or directories.

---

# GITHUB DATA SAFETY

The GitHub repository should contain reproducible engineering
artifacts and documentation.

Do NOT upload:

- WAV files
- M4A files
- MP3 files
- FLAC files
- raw recordings
- private recordings
- generated audio
- credentials
- API keys
- .env files
- virtual environments
- unnecessary large binary files
- model checkpoints

Respect the repository .gitignore.

NEVER bypass .gitignore merely to make the repository look complete.

---

# DATASET INTEGRITY

The REAL dataset is protected.

Never:

- rename REAL files
- delete REAL files
- reconvert REAL files
- modify REAL metadata
- change REAL labels
- change REAL splits

Never alter provenance.

Never invent unavailable metadata.

---

# CURRENT LANGUAGE SCOPE

Dataset V1 language scope is:

- Hindi
- Marathi

Indian English is excluded/deferred.

Do not add Indian English to Dataset V1.

---

# AUTHENTICITY LABELS

The canonical authenticity labels are:

- bonafide
- spoof

Noise, reverberation, replay, microphone, environment, distance,
and similar properties are CONDITIONS, not authenticity labels.

Never change a label merely because a recording has been processed
under a robustness condition.

---

# FILE CONTENT POLICY

If an existing file contains code that appears old, incomplete,
experimental, or questionable:

DO NOT MODIFY IT.

Instead report it as:

REQUIRES REVIEW

and wait for explicit instruction.

Do not infer that something is obsolete merely from its filename.

---

# REPOSITORY STRUCTURE

When migrating files, preserve the existing implementation.

Use the repository structure that has been explicitly approved by
the project lead.

Do not create artificial files simply to populate directories.

Do not create placeholder implementations.

Do not create fake README sections claiming functionality that does
not exist.

---

# COMMIT POLICY

Before committing:

1. Show the exact files that will be committed.
2. Verify that protected audio/data is not included.
3. Verify no credentials or secrets are included.
4. Verify existing source files were not modified.
5. Verify no unexpected generated files are included.
6. Show `git diff --cached --stat`.
7. Show `git status`.

Only then create the commit.

---

# COMMIT CONTENT

A migration commit must contain ONLY:

- already-existing project files
- approved documentation
- approved configuration
- approved scripts
- approved tests
- approved metadata/reports that are safe to publish

No newly generated implementation.

---

# COMMIT MESSAGE

Use a descriptive migration commit message such as:

chore: migrate existing VoiceGuard project artifacts

Do not use:

- final
- updated
- changes
- stuff
- test

---

# PUSH POLICY

Push ONLY to the intended VoiceGuard GitHub repository.

Do NOT:

- force push
- rewrite history
- delete branches
- push to another repository

If the remote repository is not:

ankoolsrivastava/voiceguard-voice-integrity

STOP and ask for confirmation.

---

# STOP CONDITIONS

STOP and ask the project lead before proceeding if:

1. A file needs modification.
2. A file needs renaming.
3. A file needs deletion.
4. A dataset file appears ambiguous.
5. A file contains sensitive information.
6. A file appears to contain credentials.
7. A file is not clearly safe to publish.
8. A large binary file is proposed for upload.
9. A file is protected by .gitignore but someone proposes overriding it.
10. Repository structure requires changing existing code.
11. GitHub remote is unexpected.
12. A merge conflict would require editing files.
13. Any action would change project functionality.

When a stop condition occurs, DO NOT solve it automatically.

Report the issue and wait for explicit instructions.

---

# SUCCESS CRITERION

The migration is successful ONLY when:

LOCAL EXISTING FILE
↓
READ
↓
APPROVED
↓
COPIED WITHOUT CONTENT CHANGES
↓
STAGED
↓
REVIEWED
↓
COMMITTED
↓
PUSHED TO GITHUB

The goal is NOT to make the project look complete.

The goal is to publish the REAL existing engineering work safely,
without changing its implementation.
