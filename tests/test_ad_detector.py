"""
Unit tests for ad_detector.py

All tests are fully self-contained: no real video files or external libraries
(opencv-python, scenedetect, scikit-learn) are required because the relevant
functions are mocked wherever they touch I/O or third-party code.
"""

import os
import sys
import tempfile
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import ad_detector
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Stub out heavy third-party imports so tests work without them installed
# ---------------------------------------------------------------------------

# --- cv2 stub ---
cv2_stub = types.ModuleType("cv2")
cv2_stub.VideoCapture = MagicMock()
cv2_stub.COLOR_BGR2HSV = 40
cv2_stub.cvtColor = lambda img, code: img
cv2_stub.calcHist = MagicMock(return_value=MagicMock())
cv2_stub.normalize = lambda src, dst, **kw: src
cv2_stub.compareHist = MagicMock(return_value=0.9)
cv2_stub.HISTCMP_CORREL = 3
sys.modules.setdefault("cv2", cv2_stub)

# --- numpy ---
import numpy as np  # noqa: E402  (real numpy is available in CI)

# --- scenedetect stubs ---
for mod in ["scenedetect", "scenedetect.detectors"]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

# --- sklearn stubs ---
for mod in ["sklearn", "sklearn.cluster"]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

# Now import the module under test (after stubs are in place)
import ad_detector  # noqa: E402


# ===========================================================================
# parse_video_timestamp
# ===========================================================================


class TestParseVideoTimestamp:
    def test_standard_underscore_format(self):
        ts = ad_detector.parse_video_timestamp("CNN_20230601_180000.mp4")
        assert ts == datetime(2023, 6, 1, 18, 0, 0)

    def test_channel_prefix_ignored(self):
        ts = ad_detector.parse_video_timestamp("BBC_20230101_235959.ts")
        assert ts == datetime(2023, 1, 1, 23, 59, 59)

    def test_hyphen_separator(self):
        ts = ad_detector.parse_video_timestamp("SKY-20230615-120000.mkv")
        assert ts == datetime(2023, 6, 15, 12, 0, 0)

    def test_no_timestamp_returns_none(self):
        assert ad_detector.parse_video_timestamp("random_video.mp4") is None

    def test_stem_only(self):
        ts = ad_detector.parse_video_timestamp("20230601_180000")
        assert ts == datetime(2023, 6, 1, 18, 0, 0)

    def test_invalid_date_returns_none(self):
        # Month 99 is invalid – strptime should raise → return None
        assert ad_detector.parse_video_timestamp("CH_20239901_000000.mp4") is None


# ===========================================================================
# format_duration
# ===========================================================================


class TestFormatDuration:
    def test_zero(self):
        assert ad_detector.format_duration(0) == "00:00:00"

    def test_one_hour(self):
        assert ad_detector.format_duration(3600) == "01:00:00"

    def test_mixed(self):
        assert ad_detector.format_duration(3661) == "01:01:01"

    def test_negative_clamped_to_zero(self):
        assert ad_detector.format_duration(-5) == "00:00:00"


# ===========================================================================
# find_relevant_videos
# ===========================================================================


class TestFindRelevantVideos:
    """Uses a temporary directory with fake files; cv2 is mocked."""

    def _make_tmpdir(self, filenames):
        tmpdir = tempfile.mkdtemp()
        for fname in filenames:
            Path(os.path.join(tmpdir, fname)).touch()
        return tmpdir

    def _mock_video_info(self, fps=25.0, duration=3600.0):
        """Patch get_video_info to return constant values."""
        return patch(
            "ad_detector.get_video_info",
            return_value=(fps, duration),
        )

    def test_finds_overlapping_video(self):
        tmpdir = self._make_tmpdir(["CNN_20230601_180000.mp4"])
        prog_start = datetime(2023, 6, 1, 18, 30)
        prog_end = datetime(2023, 6, 1, 19, 0)
        with self._mock_video_info(duration=3600):
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert len(results) == 1
        assert results[0]["filename"] == "CNN_20230601_180000.mp4"

    def test_ignores_wrong_channel(self):
        tmpdir = self._make_tmpdir(["BBC_20230601_180000.mp4"])
        prog_start = datetime(2023, 6, 1, 18, 30)
        prog_end = datetime(2023, 6, 1, 19, 0)
        with self._mock_video_info(duration=3600):
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert results == []

    def test_ignores_non_overlapping_video(self):
        # Video ends before programme starts.
        tmpdir = self._make_tmpdir(["CNN_20230601_150000.mp4"])
        prog_start = datetime(2023, 6, 1, 18, 0)
        prog_end = datetime(2023, 6, 1, 19, 0)
        with self._mock_video_info(duration=1800):  # 30-min video → ends 15:30
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert results == []

    def test_multiple_videos_spanning_program(self):
        tmpdir = self._make_tmpdir(
            ["CNN_20230601_180000.mp4", "CNN_20230601_183000.mp4"]
        )
        prog_start = datetime(2023, 6, 1, 18, 0)
        prog_end = datetime(2023, 6, 1, 19, 30)
        with self._mock_video_info(duration=3600):
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert len(results) == 2

    def test_ignores_unknown_extension(self):
        tmpdir = self._make_tmpdir(["CNN_20230601_180000.xyz"])
        prog_start = datetime(2023, 6, 1, 18, 30)
        prog_end = datetime(2023, 6, 1, 19, 0)
        with self._mock_video_info(duration=3600):
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert results == []

    def test_ignores_video_with_zero_duration(self):
        tmpdir = self._make_tmpdir(["CNN_20230601_180000.mp4"])
        prog_start = datetime(2023, 6, 1, 18, 30)
        prog_end = datetime(2023, 6, 1, 19, 0)
        with self._mock_video_info(fps=0.0, duration=0.0):
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert results == []

    def test_channel_match_is_case_insensitive(self):
        tmpdir = self._make_tmpdir(["cnn_20230601_180000.mp4"])
        prog_start = datetime(2023, 6, 1, 18, 30)
        prog_end = datetime(2023, 6, 1, 19, 0)
        with self._mock_video_info(duration=3600):
            results = ad_detector.find_relevant_videos(
                tmpdir, "CNN", prog_start, prog_end
            )
        assert len(results) == 1


# ===========================================================================
# merge_ad_scenes
# ===========================================================================


class TestMergeAdScenes:
    def test_no_ads(self):
        scenes = [(0, 10), (10, 20), (20, 30)]
        flags = [False, False, False]
        assert ad_detector.merge_ad_scenes(scenes, flags) == []

    def test_all_ads_merged_into_one_block(self):
        scenes = [(0, 10), (10, 20), (20, 30)]
        flags = [True, True, True]
        blocks = ad_detector.merge_ad_scenes(scenes, flags)
        assert blocks == [(0, 30)]

    def test_separate_ad_blocks(self):
        scenes = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
        flags = [True, False, False, False, True]
        blocks = ad_detector.merge_ad_scenes(scenes, flags)
        assert len(blocks) == 2
        assert blocks[0] == (0, 10)
        assert blocks[1] == (40, 50)

    def test_gap_tolerance_merges_nearby_ad_blocks(self):
        # Ad at 0–10, gap of 1 s, ad at 11–20 → should merge (gap_tolerance=2)
        scenes = [(0, 10), (10, 11), (11, 20)]
        flags = [True, False, True]
        blocks = ad_detector.merge_ad_scenes(scenes, flags, gap_tolerance=2.0)
        assert len(blocks) == 1
        assert blocks[0][0] == 0
        assert blocks[0][1] == 20

    def test_gap_exceeds_tolerance_keeps_separate(self):
        scenes = [(0, 10), (10, 15), (15, 25)]
        flags = [True, False, True]
        blocks = ad_detector.merge_ad_scenes(scenes, flags, gap_tolerance=2.0)
        assert len(blocks) == 2

    def test_empty_input(self):
        assert ad_detector.merge_ad_scenes([], []) == []


# ===========================================================================
# classify_ad_scenes  (mocked histogram extraction)
# ===========================================================================


class TestClassifyAdScenes:
    def _make_hists(self, n_program, n_ad):
        """Return (program hists, ad hists) as 1-D float32 arrays.

        The histogram has 144 bins = 18 H-bins × 8 S-bins (matching
        extract_scene_histogram).  Programme scenes concentrate energy in the
        first 72 bins (low hue); ad scenes concentrate in the upper 72 bins
        (high hue), making the two groups visually separable for clustering.
        """
        prog_hist = np.zeros(144, dtype=np.float32)
        prog_hist[:72] = 1.0 / 72  # mostly low hue
        ad_hist = np.zeros(144, dtype=np.float32)
        ad_hist[72:] = 1.0 / 72   # mostly high hue
        return (
            [prog_hist.copy() for _ in range(n_program)],
            [ad_hist.copy() for _ in range(n_ad)],
        )

    def test_too_few_scenes_returns_all_false(self):
        with patch("ad_detector.extract_scene_histogram", return_value=None):
            scenes = [(0, 10)]
            flags = ad_detector.classify_ad_scenes("fake.mp4", scenes)
        assert flags == [False]

    def test_kmeans_path_identifies_ad_cluster(self):
        prog_hists, ad_hists = self._make_hists(n_program=6, n_ad=2)
        all_hists = prog_hists + ad_hists  # 8 total; ad cluster is smaller

        scenes = [(i * 10, (i + 1) * 10) for i in range(8)]
        # Simulate KMeans: 0 = program cluster (6 items), 1 = ad cluster (2 items).
        mock_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        with patch("ad_detector.extract_scene_histogram", side_effect=all_hists), \
             patch("ad_detector._kmeans_labels", return_value=mock_labels):
            flags = ad_detector.classify_ad_scenes("fake.mp4", scenes,
                                                   diff_threshold=0.35)
        # The 2 "ad" scenes (indices 6 and 7) must be flagged.
        assert any(flags[6:])
        assert not any(flags[:6])

    def test_long_scenes_not_flagged_as_ads(self):
        """Scenes longer than MAX_PROGRAM_SCENE_DURATION are forced to False."""
        # Build one "outlier" histogram that would normally be flagged.
        prog_hist = np.zeros(144, dtype=np.float32)
        prog_hist[:72] = 1.0 / 72
        ad_hist = np.zeros(144, dtype=np.float32)
        ad_hist[72:] = 1.0 / 72

        hists = [prog_hist] * 5 + [ad_hist]
        scenes = [(i * 10, (i + 1) * 10) for i in range(5)]
        # Last scene is very long (> MAX_PROGRAM_SCENE_DURATION)
        scenes.append((50, 50 + ad_detector.MAX_PROGRAM_SCENE_DURATION + 10))

        with patch("ad_detector.extract_scene_histogram", side_effect=hists):
            flags = ad_detector.classify_ad_scenes("fake.mp4", scenes)

        assert flags[-1] is False  # long scene must NOT be marked as ad


# ===========================================================================
# write_report
# ===========================================================================


class TestWriteReport:
    def _run(self, ad_blocks):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as fh:
            path = fh.name

        ad_detector.write_report(
            path,
            program_name="Evening News",
            channel_name="CNN",
            program_start=datetime(2023, 6, 1, 18, 0, 0),
            program_end=datetime(2023, 6, 1, 19, 0, 0),
            ad_blocks=ad_blocks,
        )
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        os.unlink(path)
        return content

    def test_header_contains_program_and_channel(self):
        content = self._run([])
        assert "Evening News" in content
        assert "CNN" in content

    def test_no_ads_message(self):
        content = self._run([])
        assert "No advertisements detected" in content

    def test_ad_block_appears_in_report(self):
        abs_start = datetime(2023, 6, 1, 18, 15, 0)
        abs_end = datetime(2023, 6, 1, 18, 17, 30)
        content = self._run([(abs_start, abs_end, "CNN_20230601_180000.mp4")])
        assert "18:15:00" in content
        assert "18:17:30" in content
        assert "CNN_20230601_180000.mp4" in content

    def test_multiple_ad_blocks(self):
        blocks = [
            (datetime(2023, 6, 1, 18, 10), datetime(2023, 6, 1, 18, 12), "v1.mp4"),
            (datetime(2023, 6, 1, 18, 30), datetime(2023, 6, 1, 18, 33), "v1.mp4"),
        ]
        content = self._run(blocks)
        assert "2 block(s)" in content


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestCLI:
    def test_missing_required_args_exits(self):
        with pytest.raises(SystemExit):
            ad_detector.main(["--folder", "/tmp"])

    def test_invalid_date_format_exits(self):
        with pytest.raises(SystemExit):
            ad_detector.main(
                [
                    "--folder", "/tmp",
                    "--channel", "CNN",
                    "--program", "News",
                    "--start", "not-a-date",
                    "--end", "20230601_190000",
                ]
            )

    def test_end_before_start_exits(self):
        with pytest.raises(SystemExit):
            ad_detector.main(
                [
                    "--folder", "/tmp",
                    "--channel", "CNN",
                    "--program", "News",
                    "--start", "20230601_190000",
                    "--end", "20230601_180000",
                ]
            )

    def test_nonexistent_folder_exits(self):
        with pytest.raises(SystemExit):
            ad_detector.main(
                [
                    "--folder", "/nonexistent_path_xyz",
                    "--channel", "CNN",
                    "--program", "News",
                    "--start", "20230601_180000",
                    "--end", "20230601_190000",
                ]
            )

    def test_valid_args_calls_process_program(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("ad_detector.process_program", return_value=[]) as mock_pp:
            ad_detector.main(
                [
                    "--folder", tmpdir,
                    "--channel", "CNN",
                    "--program", "Evening News",
                    "--start", "20230601_180000",
                    "--end", "20230601_190000",
                    "--output", os.path.join(tmpdir, "out.txt"),
                ]
            )
        mock_pp.assert_called_once()
        call_kwargs = mock_pp.call_args
        assert call_kwargs.kwargs["channel_name"] == "CNN"
        assert call_kwargs.kwargs["program_name"] == "Evening News"
        assert call_kwargs.kwargs["program_start"] == datetime(2023, 6, 1, 18, 0)
        assert call_kwargs.kwargs["program_end"] == datetime(2023, 6, 1, 19, 0)
