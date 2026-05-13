# video_classifier

A command-line tool that scans a folder of recorded TV channel videos, detects
advertisement breaks during a user-specified programme window, and writes an
annotated report to a text file.

---

## How it works

1. **Discover** – Scans the given folder for video files whose filename
   contains the channel name and whose recording window overlaps the
   programme timeline.

2. **Detect scenes** – Runs
   [PySceneDetect](https://www.scenedetect.com/) `ContentDetector` on the
   portion of each video that overlaps the programme window to find all
   scene transitions.

3. **Classify** – Extracts a colour-histogram "signature" for every scene.
   KMeans (k = 2) splits scenes into two visual clusters; the smaller cluster
   is treated as potential advertisements.  Very long scenes
   (> 5 minutes) are always treated as programme content.

4. **Merge** – Consecutive ad scenes (with configurable gap tolerance) are
   merged into contiguous ad blocks.

5. **Report** – A `.txt` file is written containing the programme metadata
   and one line per detected ad block, showing absolute timestamps, duration,
   and offset within the programme.

---

## Video filename convention

```
{CHANNEL}_{YYYYMMDD}_{HHMMSS}.{ext}
```

Examples:

```
CNN_20230601_180000.mp4
BBC_20230601_183000.ts
SKY_20230601_190000.mkv
```

Supported extensions: `.mp4`, `.avi`, `.mkv`, `.ts`, `.mov`, `.m2ts`,
`.mpg`, `.mpeg`, `.wmv`.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `numpy`, `scenedetect`, `scikit-learn`.

---

## Usage

```
python ad_detector.py \
    --folder   /path/to/recordings \
    --channel  CNN \
    --program  "Evening News" \
    --start    20230601_180000 \
    --end      20230601_190000 \
    --output   evening_news_ads.txt
```

### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--folder` | ✓ | – | Folder containing the video files |
| `--channel` | ✓ | – | Channel name substring (case-insensitive, matched against filenames) |
| `--program` | ✓ | – | Programme name (written to the report) |
| `--start` | ✓ | – | Programme start time: `YYYYMMDD_HHMMSS` |
| `--end` | ✓ | – | Programme end time: `YYYYMMDD_HHMMSS` |
| `--output` | | `ad_report.txt` | Output report file |
| `--scene-threshold` | | `27.0` | ContentDetector threshold (lower → more sensitive) |
| `--diff-threshold` | | `0.35` | Visual-difference threshold for ad classification (0–1) |
| `--gap-tolerance` | | `2.0` | Max gap (seconds) between ad scenes to merge into one block |

---

## Sample output (`ad_report.txt`)

```
Program  : Evening News
Channel  : CNN
Start    : 2023-06-01 18:00:00
End      : 2023-06-01 19:00:00
Duration : 01:00:00

Detected Ads: 3 block(s)
------------------------------------------------------------
Ad   1: 2023-06-01 18:12:00 – 2023-06-01 18:14:30  | duration 00:02:30  | programme offset 00:12:00–00:14:30  | source: CNN_20230601_180000.mp4
Ad   2: 2023-06-01 18:28:15 – 2023-06-01 18:31:00  | duration 00:02:45  | programme offset 00:28:15–00:31:00  | source: CNN_20230601_180000.mp4
Ad   3: 2023-06-01 18:47:40 – 2023-06-01 18:50:10  | duration 00:02:30  | programme offset 00:47:40–00:50:10  | source: CNN_20230601_183000.mp4
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
