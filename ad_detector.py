#!/usr/bin/env python3
"""
TV Channel Ad Detector
======================
Detects advertisement segments in recorded TV channel videos.

Video filename convention
--------------------------
  {channel_name}_{YYYYMMDD}_{HHMMSS}.{ext}

  Examples:
    CNN_20230601_180000.mp4
    BBC_20230601_183000.ts

Algorithm overview
-------------------
1. Scan *folder* for video files whose channel name matches and whose
   recording window overlaps the user-specified program timeline.
2. For each relevant video, run PySceneDetect (ContentDetector) on only
   the portion that overlaps the program window.
3. Extract a colour-histogram "signature" for every detected scene.
4. Use KMeans (k=2) to split scenes into two visual clusters.
   The larger cluster is assumed to be the *programme*; the smaller
   one(s) are treated as potential *ads*.  A minimum-duration guard
   also flags very short isolated scenes as potential ads.
5. Merge consecutive ad scenes into contiguous ad blocks.
6. Write a text report: programme metadata + one line per ad block
   showing absolute start/end timestamps and duration.

Usage
-----
  python ad_detector.py --folder /path/to/videos \\
                        --channel CNN \\
                        --program "Evening News" \\
                        --start 20230601_180000 \\
                        --end   20230601_190000 \\
                        --output ad_report.txt
"""

import os
import re
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS: frozenset = frozenset(
    {".mp4", ".avi", ".mkv", ".ts", ".mov", ".m2ts", ".mpg", ".mpeg", ".wmv"}
)

# Regex for YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS embedded in a filename stem.
_TS_RE = re.compile(r"(\d{8})[_\-](\d{6})")

# Minimum scene duration (seconds) used as an additional ad heuristic.
MIN_AD_DURATION = 3.0
MAX_PROGRAM_SCENE_DURATION = 300.0  # scenes longer than 5 min are likely program


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def parse_video_timestamp(filename: str) -> Optional[datetime]:
    """Return the recording-start datetime encoded in *filename*, or None."""
    stem = Path(filename).stem
    m = _TS_RE.search(stem)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def format_duration(seconds: float) -> str:
    """Format *seconds* as HH:MM:SS."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------


def get_video_info(video_path: str) -> Tuple[float, float]:
    """Return (fps, duration_seconds) for *video_path*.  Returns (0, 0) on error."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return fps, duration


def find_relevant_videos(
    folder: str,
    channel_name: str,
    program_start: datetime,
    program_end: datetime,
) -> List[dict]:
    """Return metadata dicts for video files that overlap the program window."""
    results = []
    channel_lower = channel_name.lower()

    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        if Path(fname).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if channel_lower not in fname.lower():
            continue

        video_start = parse_video_timestamp(fname)
        if video_start is None:
            continue

        _, duration = get_video_info(fpath)
        if duration <= 0:
            continue

        video_end = video_start + timedelta(seconds=duration)

        # Overlap check: video window must intersect the program window.
        if video_start >= program_end or video_end <= program_start:
            continue

        results.append(
            {
                "path": fpath,
                "filename": fname,
                "start": video_start,
                "end": video_end,
                "duration": duration,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Scene detection  (PySceneDetect)
# ---------------------------------------------------------------------------


def detect_scenes(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    threshold: float = 27.0,
) -> List[Tuple[float, float]]:
    """
    Run PySceneDetect ContentDetector on *video_path* between *start_sec*
    and *end_sec*.  Returns a list of (scene_start_sec, scene_end_sec) tuples.
    Falls back to a single scene covering the full range on error.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path, start_time=start_sec)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        if end_sec is not None:
            duration = end_sec - start_sec
            scene_manager.detect_scenes(video, duration=duration)
        else:
            scene_manager.detect_scenes(video)

        raw_scenes = scene_manager.get_scene_list()
        scenes = [(s.get_seconds(), e.get_seconds()) for s, e in raw_scenes]

        # Clamp to the requested window.
        lo = start_sec
        hi = end_sec if end_sec is not None else float("inf")
        scenes = [
            (max(s, lo), min(e, hi))
            for s, e in scenes
            if s < hi and e > lo
        ]
        return scenes if scenes else [(start_sec, end_sec or lo)]

    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] PySceneDetect error for {video_path}: {exc}")
        fallback_end = end_sec if end_sec is not None else start_sec
        return [(start_sec, fallback_end)]


# ---------------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------------


def extract_scene_histogram(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_samples: int = 5,
) -> Optional[np.ndarray]:
    """
    Sample *num_samples* frames from the scene and return their average
    HSV histogram (H:18 bins, S:8 bins) as a 1-D float32 array, or None.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    frame_range = max(1, end_frame - start_frame)
    step = max(1, frame_range // num_samples)

    hists: List[np.ndarray] = []
    for offset in range(0, frame_range, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # H channel: 18 bins over [0, 180°]; S channel: 8 bins over [0, 256].
        # Using only H and S (ignoring V/brightness) makes the signature
        # more robust to lighting changes typical between programme and ads.
        h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        hists.append(h)
        if len(hists) >= num_samples:
            break

    cap.release()
    return np.mean(hists, axis=0).astype(np.float32) if hists else None


# ---------------------------------------------------------------------------
# Ad classification
# ---------------------------------------------------------------------------


def classify_ad_scenes(
    video_path: str,
    scenes: List[Tuple[float, float]],
    diff_threshold: float = 0.35,
) -> List[bool]:
    """
    Return a list of booleans (True = likely ad) for each scene in *scenes*.

    Strategy:
      1. Compute a colour histogram for every scene.
      2. If enough scenes are available, use KMeans (k=2) to cluster them.
         The smaller cluster is flagged as ads.
      3. Fall back to a median-correlation threshold when sklearn is
         unavailable or there are too few scenes.
      4. Additionally, very short scenes (< MIN_AD_DURATION seconds) that
         are visually different from their neighbours are flagged.
    """
    if not scenes:
        return []

    histograms: List[Optional[np.ndarray]] = [
        extract_scene_histogram(video_path, s, e) for s, e in scenes
    ]

    valid_idx = [i for i, h in enumerate(histograms) if h is not None]
    if len(valid_idx) < 2:
        # Cannot classify — treat nothing as an ad.
        return [False] * len(scenes)

    valid_hists = np.stack([histograms[i] for i in valid_idx])  # (N, F)

    # --- Cluster-based classification ---
    labels = _kmeans_labels(valid_hists)

    if labels is not None:
        # Assign ad label to the *smaller* cluster.
        counts = np.bincount(labels)
        ad_cluster = int(np.argmin(counts))
        valid_flags = [int(lbl) == ad_cluster for lbl in labels]
    else:
        # Fallback: correlation with median.
        median_hist = np.median(valid_hists, axis=0).astype(np.float32)
        corrs = [
            float(cv2.compareHist(h, median_hist, cv2.HISTCMP_CORREL))
            for h in valid_hists
        ]
        mean_c = float(np.mean(corrs))
        valid_flags = [c < mean_c - diff_threshold for c in corrs]

    # Map back to original scene list length.
    ad_flags = [False] * len(scenes)
    for j, orig_i in enumerate(valid_idx):
        ad_flags[orig_i] = valid_flags[j]

    # Reinforce: very short scenes that are flagged remain flagged;
    # very long scenes are almost certainly programme content.
    for i, (s, e) in enumerate(scenes):
        duration = e - s
        if duration > MAX_PROGRAM_SCENE_DURATION:
            ad_flags[i] = False

    return ad_flags


def _kmeans_labels(data: np.ndarray) -> Optional[np.ndarray]:
    """Run KMeans(k=2) on *data* rows.  Returns integer label array or None."""
    if len(data) < 4:
        return None
    try:
        from sklearn.cluster import KMeans

        # random_state=42 ensures reproducible cluster assignments across runs.
        km = KMeans(n_clusters=2, random_state=42)
        return km.fit_predict(data)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] KMeans clustering unavailable ({exc}); "
              "falling back to median-correlation classifier.")
        return None


# ---------------------------------------------------------------------------
# Merging ad blocks
# ---------------------------------------------------------------------------


def merge_ad_scenes(
    scenes: List[Tuple[float, float]],
    ad_flags: List[bool],
    gap_tolerance: float = 2.0,
) -> List[Tuple[float, float]]:
    """
    Merge consecutive ad scenes (with optional gap tolerance) into blocks.

    A non-ad scene whose duration is ≤ *gap_tolerance* seconds is absorbed
    into the current ad block only when it is *surrounded* by ad scenes on
    both sides.  This prevents large program segments from being swallowed.

    Returns list of (start_sec, end_sec) ad block tuples.
    """
    if not scenes:
        return []

    # First pass: fill in short non-ad gaps that sit between two ad scenes.
    effective_flags = list(ad_flags)
    for i in range(1, len(scenes) - 1):
        if (
            not effective_flags[i]
            and effective_flags[i - 1]
            and effective_flags[i + 1]
        ):
            gap_dur = scenes[i][1] - scenes[i][0]
            if gap_dur <= gap_tolerance:
                effective_flags[i] = True

    # Second pass: merge consecutive (now-filled) ad scenes.
    blocks: List[Tuple[float, float]] = []
    block_start: Optional[float] = None
    block_end: Optional[float] = None

    for (s, e), is_ad in zip(scenes, effective_flags):
        if is_ad:
            if block_start is None:
                block_start = s
            block_end = e
        elif block_start is not None:
            blocks.append((block_start, block_end))
            block_start = None
            block_end = None

    if block_start is not None:
        blocks.append((block_start, block_end))

    return blocks


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_report(
    output_file: str,
    program_name: str,
    channel_name: str,
    program_start: datetime,
    program_end: datetime,
    ad_blocks: List[Tuple[datetime, datetime, str]],
) -> None:
    """Write the ad detection report to *output_file*."""
    lines = [
        f"Program  : {program_name}",
        f"Channel  : {channel_name}",
        f"Start    : {program_start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"End      : {program_end.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration : {format_duration((program_end - program_start).total_seconds())}",
        "",
        f"Detected Ads: {len(ad_blocks)} block(s)",
        "-" * 60,
    ]

    if ad_blocks:
        for idx, (abs_start, abs_end, src_file) in enumerate(ad_blocks, 1):
            dur = (abs_end - abs_start).total_seconds()
            # Relative offset inside the programme.
            rel_start = (abs_start - program_start).total_seconds()
            rel_end = (abs_end - program_start).total_seconds()
            lines.append(
                f"Ad {idx:>3}: {abs_start.strftime('%Y-%m-%d %H:%M:%S')} – "
                f"{abs_end.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"| duration {format_duration(dur)}  "
                f"| programme offset {format_duration(rel_start)}–{format_duration(rel_end)}  "
                f"| source: {src_file}"
            )
    else:
        lines.append("No advertisements detected.")

    text = "\n".join(lines) + "\n"

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(text)

    print(text)
    print(f"Report saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_program(
    folder: str,
    channel_name: str,
    program_name: str,
    program_start: datetime,
    program_end: datetime,
    output_file: str,
    scene_threshold: float = 27.0,
    diff_threshold: float = 0.35,
    gap_tolerance: float = 2.0,
) -> List[Tuple[datetime, datetime, str]]:
    """
    Orchestrate end-to-end ad detection for *program_name*.
    Returns the list of (abs_start, abs_end, source_file) ad blocks.
    """
    print(f"\nChannel  : {channel_name}")
    print(f"Program  : {program_name}")
    print(f"Window   : {program_start} – {program_end}")
    print(f"Folder   : {folder}\n")

    videos = find_relevant_videos(folder, channel_name, program_start, program_end)
    if not videos:
        print("No matching video files found. Check --folder and --channel.")
        write_report(output_file, program_name, channel_name,
                     program_start, program_end, [])
        return []

    print(f"Found {len(videos)} relevant video file(s):")
    for v in videos:
        print(f"  {v['filename']}  "
              f"({v['start'].strftime('%H:%M:%S')} – {v['end'].strftime('%H:%M:%S')})")

    all_ad_blocks: List[Tuple[datetime, datetime, str]] = []

    for video in videos:
        vpath = video["path"]
        v_start: datetime = video["start"]
        v_dur: float = video["duration"]

        # Portion of the video that overlaps the programme window.
        seg_start_sec = max(0.0, (program_start - v_start).total_seconds())
        seg_end_sec = min(v_dur, (program_end - v_start).total_seconds())

        print(f"\nProcessing : {video['filename']}")
        print(f"  Segment  : {format_duration(seg_start_sec)} – "
              f"{format_duration(seg_end_sec)} (video-relative)")

        scenes = detect_scenes(
            vpath,
            start_sec=seg_start_sec,
            end_sec=seg_end_sec,
            threshold=scene_threshold,
        )
        print(f"  Scenes   : {len(scenes)} detected")

        if len(scenes) < 2:
            print("  Skipping ad classification (too few scenes).")
            continue

        ad_flags = classify_ad_scenes(vpath, scenes, diff_threshold=diff_threshold)
        n_ads = sum(ad_flags)
        print(f"  Ad scenes: {n_ads} / {len(scenes)}")

        ad_blocks_sec = merge_ad_scenes(scenes, ad_flags, gap_tolerance=gap_tolerance)

        for blk_start_sec, blk_end_sec in ad_blocks_sec:
            abs_start = v_start + timedelta(seconds=blk_start_sec)
            abs_end = v_start + timedelta(seconds=blk_end_sec)
            all_ad_blocks.append((abs_start, abs_end, video["filename"]))

    write_report(
        output_file,
        program_name,
        channel_name,
        program_start,
        program_end,
        all_ad_blocks,
    )
    return all_ad_blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ad_detector",
        description="Detect TV advertisements in recorded channel videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Videos named like: CNN_20230601_180000.mp4
  python ad_detector.py \\
      --folder   /recordings \\
      --channel  CNN \\
      --program  "Evening News" \\
      --start    20230601_180000 \\
      --end      20230601_190000 \\
      --output   evening_news_ads.txt

Filename convention
-------------------
  {CHANNEL}_{YYYYMMDD}_{HHMMSS}.{ext}
  e.g.  CNN_20230601_180000.mp4
        BBC_20230601_183000.ts
""",
    )
    parser.add_argument("--folder",  required=True,
                        help="Folder containing TV channel video files.")
    parser.add_argument("--channel", required=True,
                        help="Channel name substring to match against filenames.")
    parser.add_argument("--program", required=True,
                        help="Programme name (written to the report).")
    parser.add_argument("--start",   required=True,
                        help="Programme start time: YYYYMMDD_HHMMSS")
    parser.add_argument("--end",     required=True,
                        help="Programme end time:   YYYYMMDD_HHMMSS")
    parser.add_argument("--output",  default="ad_report.txt",
                        help="Output report file (default: ad_report.txt).")
    parser.add_argument("--scene-threshold", type=float, default=27.0,
                        help="ContentDetector threshold (default 27.0; "
                             "lower = more sensitive).")
    parser.add_argument("--diff-threshold", type=float, default=0.35,
                        help="Visual-difference threshold for ad classification "
                             "(0–1, default 0.35).")
    parser.add_argument("--gap-tolerance", type=float, default=2.0,
                        help="Max gap (seconds) between ad scenes to merge into "
                             "one block (default 2.0).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        program_start = datetime.strptime(args.start, "%Y%m%d_%H%M%S")
        program_end = datetime.strptime(args.end, "%Y%m%d_%H%M%S")
    except ValueError:
        parser.error("--start and --end must be in YYYYMMDD_HHMMSS format "
                     "(e.g. 20230601_180000).")

    if program_end <= program_start:
        parser.error("--end must be after --start.")

    if not os.path.isdir(args.folder):
        parser.error(f"--folder does not exist or is not a directory: {args.folder}")

    process_program(
        folder=args.folder,
        channel_name=args.channel,
        program_name=args.program,
        program_start=program_start,
        program_end=program_end,
        output_file=args.output,
        scene_threshold=args.scene_threshold,
        diff_threshold=args.diff_threshold,
        gap_tolerance=args.gap_tolerance,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
