#!/usr/bin/env python3
"""
Extract video segments at 1080p from a source video based on timestamps in an Excel file.
Resolves exact frame numbers from DJI SRT files and verifies them.

Excel format:
    Column 1: video path
    Column 2: SRT path
    Columns 3+: alternating start/end timestamps (HH:MM:SS or MM:SS)
    e.g., video_path | srt_path | start1 | end1 | start2 | end2 | ...

Output structure:
    <video_parent_dir>/<video_stem>/
        seg1.mp4  +  seg1_metadata.json
        seg2.mp4  +  seg2_metadata.json
        ...

Usage:
    python scripts/extract_segments.py --excel "/mnt/c/Users/shukl/Downloads/extract1k.xlsx"
    python scripts/extract_segments.py --excel input.xlsx --output-res 1280x720
"""

import argparse
import bisect
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video segments at 1080p from timestamps in an Excel file"
    )
    parser.add_argument(
        "--excel", type=str, required=True,
        help="Path to the Excel file with video paths, SRT paths, and timestamps",
    )
    parser.add_argument(
        "--output-res", type=str, default="1920:1080",
        help="Output resolution as WxH or W:H (default: 1920:1080 i.e. 1080p)",
    )
    return parser.parse_args()


def win_to_wsl_path(path: str) -> str:
    """Convert a Windows path (e.g. C:\\Users\\...) to WSL path (/mnt/c/Users/...)."""
    path = path.strip()
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path


def normalize_timestamp(ts: str) -> str:
    """
    Normalize a timestamp string to HH:MM:SS format for ffmpeg.
    Accepts: M:SS, MM:SS, H:MM:SS, HH:MM:SS
    """
    ts = str(ts).strip()
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00:{int(parts[0]):02d}:{int(parts[1]):02d}"
    elif len(parts) == 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    else:
        raise ValueError(f"Cannot parse timestamp: '{ts}'")


def timestamp_to_ms(ts: str) -> int:
    """Convert HH:MM:SS to milliseconds."""
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return (h * 3600 + m * 60 + s) * 1000


def timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS to seconds."""
    return timestamp_to_ms(ts) / 1000.0


# ---------------------------------------------------------------------------
# SRT Parsing
# ---------------------------------------------------------------------------

def parse_srt_timestamps(srt_path: str) -> Optional[List[Tuple[int, int]]]:
    """
    Parse a DJI SRT file and extract (start_ms, framecnt) for each entry.

    DJI SRT entries look like:
        1
        00:00:00,000 --> 00:00:00,033
        FrameCnt: 1, DiffTime: 33ms ...

    Returns a sorted list of (start_ms, framecnt) tuples, or None on failure.
    """
    if not os.path.isfile(srt_path):
        print(f"  SRT file not found: {srt_path}")
        return None

    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split into blocks (separated by blank lines)
    blocks = re.split(r"\n\s*\n", content)

    # Regex for SRT timestamp line: HH:MM:SS,mmm --> HH:MM:SS,mmm
    ts_re = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->")
    # Regex for FrameCnt or SrtCnt
    frame_re = re.compile(r"(?:FrameCnt|SrtCnt)\s*:\s*(\d+)")

    entries = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        ts_match = ts_re.search(block)
        frame_match = frame_re.search(block)

        if ts_match and frame_match:
            h, m, s, ms = (
                int(ts_match.group(1)),
                int(ts_match.group(2)),
                int(ts_match.group(3)),
                int(ts_match.group(4)),
            )
            start_ms = h * 3600000 + m * 60000 + s * 1000 + ms
            framecnt = int(frame_match.group(1))
            entries.append((start_ms, framecnt))

    if not entries:
        return None

    # Sort by timestamp (should already be sorted, but be safe)
    entries.sort(key=lambda x: x[0])
    return entries


def lookup_frame_range(
    srt_entries: List[Tuple[int, int]], start_time: str, end_time: str
) -> Tuple[int, int]:
    """
    Find exact FrameCnt values for start/end timestamps using binary search.

    start_time (HH:MM:SS) → first SRT entry at or after that second
    end_time (HH:MM:SS)   → last SRT entry within that second
    """
    start_ms = timestamp_to_ms(start_time)
    end_ms = timestamp_to_ms(end_time) + 999  # last millisecond of that second

    # Extract just the ms values for binary search
    ms_values = [e[0] for e in srt_entries]

    # First entry at or after start_ms
    start_idx = bisect.bisect_left(ms_values, start_ms)
    if start_idx >= len(srt_entries):
        start_idx = len(srt_entries) - 1

    # Last entry at or before end_ms
    end_idx = bisect.bisect_right(ms_values, end_ms) - 1
    if end_idx < 0:
        end_idx = 0

    start_frame = srt_entries[start_idx][1]
    end_frame = srt_entries[end_idx][1]

    return start_frame, end_frame


# ---------------------------------------------------------------------------
# Video info & verification
# ---------------------------------------------------------------------------

def get_video_fps(video_path: str) -> Optional[float]:
    """Get exact FPS from video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    # Parse fractional FPS like "30000/1001"
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


def count_segment_frames(segment_path: str) -> Optional[int]:
    """Count actual frames in an extracted segment using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        segment_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segment(
    video_path: str,
    start: str,
    end: str,
    output_path: str,
    width: int,
    height: int,
) -> bool:
    """Run ffmpeg to extract a segment scaled to the target resolution."""
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start,
        "-i", video_path,
        "-to", end,
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: ffmpeg failed: {result.stderr.strip()}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse resolution
    res = args.output_res.replace("x", ":").split(":")
    if len(res) != 2:
        print(f"Invalid resolution format: {args.output_res}. Use WxH or W:H.")
        sys.exit(1)
    width, height = int(res[0]), int(res[1])

    # Read Excel
    excel_path = args.excel
    if not os.path.isfile(excel_path):
        print(f"Excel file not found: {excel_path}")
        sys.exit(1)

    df = pd.read_excel(excel_path, header=None)
    print(f"Read {len(df)} rows from {excel_path}\n")

    total_segments = 0
    total_errors = 0

    for row_idx, row in df.iterrows():
        video_path = win_to_wsl_path(str(row.iloc[0]).strip())
        srt_path = win_to_wsl_path(str(row.iloc[1]).strip())

        if not os.path.isfile(video_path):
            print(f"[Row {row_idx + 1}] Video not found, skipping: {video_path}")
            total_errors += 1
            continue

        # Get video FPS
        fps = get_video_fps(video_path)
        if fps is None:
            print(f"[Row {row_idx + 1}] Cannot read FPS, skipping: {video_path}")
            total_errors += 1
            continue

        # Parse SRT
        srt_entries = None
        if os.path.isfile(srt_path):
            srt_entries = parse_srt_timestamps(srt_path)

        # Collect non-empty timestamp values from columns 2+ (0=video, 1=srt, 2+=timestamps)
        timestamps = []
        for col_idx in range(2, len(row)):
            val = row.iloc[col_idx]
            if pd.isna(val) or str(val).strip() == "":
                break
            timestamps.append(str(val).strip())

        if len(timestamps) < 2 or len(timestamps) % 2 != 0:
            print(
                f"[Row {row_idx + 1}] Invalid number of timestamps ({len(timestamps)}), "
                "need even count of start/end pairs. Skipping."
            )
            total_errors += 1
            continue

        # Create output folder named after the video
        video_stem = Path(video_path).stem
        video_parent = Path(video_path).parent
        output_dir = video_parent / video_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        num_segments = len(timestamps) // 2
        srt_info = f", SRT: {len(srt_entries)} entries" if srt_entries else ", SRT: not found"
        print(f"[Row {row_idx + 1}] {video_stem}: {num_segments} segment(s), {fps:.2f} fps{srt_info}")

        for i in range(num_segments):
            seg_num = i + 1
            raw_start = timestamps[2 * i]
            raw_end = timestamps[2 * i + 1]

            try:
                start = normalize_timestamp(raw_start)
                end = normalize_timestamp(raw_end)
            except ValueError as e:
                print(f"  seg{seg_num}: {e}, skipping")
                total_errors += 1
                continue

            # Resolve frame numbers
            start_frame, end_frame = None, None
            fps_start, fps_end = None, None

            # FPS-based calculation
            fps_start = round(timestamp_to_seconds(start) * fps)
            fps_end = round((timestamp_to_seconds(end) + 1) * fps) - 1  # last frame of end second

            # SRT-based lookup (primary, if available)
            if srt_entries:
                start_frame, end_frame = lookup_frame_range(srt_entries, start, end)
            else:
                start_frame, end_frame = fps_start, fps_end

            total_frames = end_frame - start_frame + 1

            # Verification: SRT vs FPS math
            verification = "ok"
            if srt_entries and fps_start is not None:
                drift = abs(start_frame - fps_start)
                if drift > 2:
                    verification = f"WARN: SRT/FPS drift={drift} frames"

            # Extract segment
            output_file = f"seg{seg_num}.mp4"
            output_path = str(output_dir / output_file)

            print(f"  seg{seg_num}: {start} -> {end} | frames {start_frame}-{end_frame} ({total_frames} frames) | {verification}")
            ok = extract_segment(video_path, start, end, output_path, width, height)

            if ok:
                total_segments += 1

                # Verify actual frame count in extracted segment
                actual_frames = count_segment_frames(output_path)
                if actual_frames is not None and abs(actual_frames - total_frames) > 2:
                    print(f"    WARN: segment has {actual_frames} frames, expected ~{total_frames}")

                # Write metadata
                metadata = {
                    "source_video": video_path,
                    "source_srt": srt_path,
                    "segment_index": seg_num,
                    "start_time": start,
                    "end_time": end,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "total_frames": total_frames,
                    "video_fps": round(fps, 4),
                    "fps_math_start_frame": fps_start,
                    "fps_math_end_frame": fps_end,
                    "actual_segment_frames": actual_frames,
                    "output_resolution": f"{width}x{height}",
                    "output_file": output_file,
                    "extracted_on": datetime.now().isoformat(),
                }
                meta_path = str(output_dir / f"seg{seg_num}_metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                total_errors += 1

        print()

    print(f"Done. Extracted {total_segments} segment(s), {total_errors} error(s).")


if __name__ == "__main__":
    main()
