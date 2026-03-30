#!/usr/bin/env python3
"""
Extract video segments as frames (JPG) from a source video based on timestamps in an Excel file.
Resolves exact frame numbers from DJI SRT files and verifies them.

Excel format:
    Column 1: video path
    Column 2: SRT path
    Columns 3+: alternating start/end timestamps (HH:MM:SS or MM:SS)
    e.g., video_path | srt_path | start1 | end1 | start2 | end2 | ...

Output structure:
    <output_dir>/<video_stem>/
        <srt_filename>.srt          (copied SRT)
        seg1/                       (frames as JPGs with original frame numbers)
            frame_000300.jpg
            frame_000303.jpg
            metadata.json
        seg2/
            ...
        progress.json               (tracks completed segments for resume)

Segments longer than --max-frames are split into multiple sequential seg folders.
Segments (or tail chunks) with fewer than 50 frames are discarded.

Resume support:
    A progress.json is maintained per video output directory. Re-running the
    same command skips completed segments, redoes the last incomplete one, and
    continues. If --extract-fps, --max-frames, or --output-res change between
    runs, progress is invalidated and the video is re-processed from scratch.

Usage:
    python scripts/extract_segments_frames.py --excel input.xlsx --output-dir /path/to/output --extract-fps 10 --max-frames 200
"""

import argparse
import bisect
import json
import os
import re
import shutil
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
        description="Extract video segments as frames from timestamps in an Excel file"
    )
    parser.add_argument(
        "--excel", type=str, required=True,
        help="Path to the Excel file with video paths, SRT paths, and timestamps",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Base output directory for extracted frames",
    )
    parser.add_argument(
        "--output-res", type=str, default="1920:1080",
        help="Output resolution as WxH or W:H (default: 1920:1080 i.e. 1080p)",
    )
    parser.add_argument(
        "--extract-fps", type=float, required=True,
        help="Frame rate at which to extract frames (e.g. 10 for 10fps)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
        help="Max extracted frames per segment folder (default: 200). "
             "Longer segments are split. Chunks with <50 frames are discarded.",
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
# Progress tracking (resume support)
# ---------------------------------------------------------------------------

PROGRESS_FILE = "progress.json"


def load_progress(video_out_dir: str, params: dict) -> dict:
    """
    Load progress.json from a video output directory.
    Returns the progress dict if params match, otherwise returns a fresh one.
    """
    progress_path = os.path.join(video_out_dir, PROGRESS_FILE)
    fresh = {
        "params": params,
        "completed_segments": {},
        "video_complete": False,
    }

    if not os.path.isfile(progress_path):
        return fresh

    with open(progress_path, "r") as f:
        progress = json.load(f)

    # Invalidate if params changed
    if progress.get("params") != params:
        print(f"  Parameters changed, restarting extraction for this video")
        return fresh

    return progress


def save_progress(video_out_dir: str, progress: dict):
    """Write progress.json atomically."""
    progress_path = os.path.join(video_out_dir, PROGRESS_FILE)
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, progress_path)


def is_segment_complete(video_out_dir: str, seg_name: str, expected_frames: int) -> bool:
    """Check if a segment folder exists on disk with the expected number of JPGs."""
    seg_dir = os.path.join(video_out_dir, seg_name)
    if not os.path.isdir(seg_dir):
        return False
    jpg_count = len([f for f in os.listdir(seg_dir) if f.endswith(".jpg")])
    return jpg_count == expected_frames


# ---------------------------------------------------------------------------
# Video info
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


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames_for_segment(
    video_path: str,
    start_time: str,
    end_time: str,
    start_frame: int,
    video_fps: float,
    extract_fps: float,
    width: int,
    height: int,
    output_dir: str,
    max_frames: int,
    global_seg_counter: int,
    pair_index: int,
    srt_relative_path: Optional[str],
    progress: dict,
) -> Tuple[int, int]:
    """
    Extract frames from a video segment, splitting into sub-folders if needed.

    Frames are extracted at extract_fps and named with their original video frame number.
    Skips segments already marked complete in progress. Redoes incomplete ones.
    Returns (next global_seg_counter, number of frames successfully extracted).
    """
    # Calculate the step in original frame numbers
    # e.g. video at 30fps, extract at 10fps → every 3rd frame
    frame_step = round(video_fps / extract_fps)
    if frame_step < 1:
        frame_step = 1

    start_sec = timestamp_to_seconds(start_time)
    end_sec = timestamp_to_seconds(end_time) + 1.0  # include the last second

    # Build list of original frame numbers to extract
    total_video_frames = round((end_sec - start_sec) * video_fps)
    end_frame_limit = start_frame + total_video_frames

    frame_numbers = list(range(start_frame, end_frame_limit, frame_step))

    if not frame_numbers:
        return global_seg_counter, 0

    # Split into chunks of max_frames
    chunks = []
    for i in range(0, len(frame_numbers), max_frames):
        chunk = frame_numbers[i:i + max_frames]
        # Discard chunks with fewer than 50 frames
        if len(chunk) < 50:
            print(f"    Discarding chunk with {len(chunk)} frames (<50)")
            continue
        chunks.append(chunk)

    total_extracted = 0
    completed_segments = progress.get("completed_segments", {})

    for chunk in chunks:
        seg_name = f"seg{global_seg_counter}"
        seg_dir = os.path.join(output_dir, seg_name)
        pair_label = f"pair{pair_index + 1}/{seg_name}"

        # --- Resume logic ---
        if seg_name in completed_segments:
            expected = completed_segments[seg_name]["num_frames"]
            if is_segment_complete(output_dir, seg_name, expected):
                print(f"    {pair_label}: SKIP (already complete, {expected} frames)")
                total_extracted += completed_segments[seg_name]["num_extracted"]
                global_seg_counter += 1
                continue
            else:
                print(f"    {pair_label}: incomplete on disk, redoing")

        # If folder exists but not in progress (incomplete), wipe and redo
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir)

        os.makedirs(seg_dir, exist_ok=True)

        print(f"    {pair_label}: {len(chunk)} frames, "
              f"original frames {chunk[0]}-{chunk[-1]} (step={frame_step})")

        extracted_in_chunk = 0
        for frame_num in chunk:
            # Calculate the timestamp for this frame
            frame_time = frame_num / video_fps
            output_file = os.path.join(seg_dir, f"frame_{frame_num:06d}.jpg")

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{frame_time:.6f}",
                "-i", video_path,
                "-vframes", "1",
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-q:v", "2",  # ~q95 for MJPEG
                "-loglevel", "error",
                output_file,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"      ERROR extracting frame {frame_num}: {result.stderr.strip()}")
            else:
                extracted_in_chunk += 1

        # Write chunk metadata
        metadata = {
            "segment_folder": seg_name,
            "source_pair": pair_index + 1,
            "pair_seg_label": pair_label,
            "srt_path": srt_relative_path,
            "source_video": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "num_frames": len(chunk),
            "num_extracted": extracted_in_chunk,
            "first_original_frame": chunk[0],
            "last_original_frame": chunk[-1],
            "frame_step": frame_step,
            "frame_numbers": chunk,
            "video_fps": round(video_fps, 4),
            "extract_fps": extract_fps,
            "output_resolution": f"{width}x{height}",
            "extracted_on": datetime.now().isoformat(),
        }
        meta_path = os.path.join(seg_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Mark segment complete in progress and save immediately
        progress["completed_segments"][seg_name] = {
            "pair_index": pair_index,
            "pair_label": pair_label,
            "num_frames": len(chunk),
            "num_extracted": extracted_in_chunk,
        }
        save_progress(output_dir, progress)

        total_extracted += extracted_in_chunk
        global_seg_counter += 1

    return global_seg_counter, total_extracted


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

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Read Excel
    excel_path = args.excel
    if not os.path.isfile(excel_path):
        print(f"Excel file not found: {excel_path}")
        sys.exit(1)

    df = pd.read_excel(excel_path, header=None)
    print(f"Read {len(df)} rows from {excel_path}")
    print(f"Settings: extract_fps={args.extract_fps}, max_frames={args.max_frames}, "
          f"resolution={width}x{height}\n")

    # Build params dict for progress validation
    run_params = {
        "extract_fps": args.extract_fps,
        "max_frames": args.max_frames,
        "resolution": f"{width}x{height}",
    }

    total_segments = 0
    total_frames_extracted = 0
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

        # Collect non-empty timestamp values from columns 2+
        # Special case: if column 2 says "whole video", process the entire video
        whole_video = False
        timestamps = []
        first_val = row.iloc[2] if len(row) > 2 else None
        if (
            first_val is not None
            and not pd.isna(first_val)
            and str(first_val).strip().lower() == "whole video"
        ):
            whole_video = True
            duration = get_video_duration(video_path)
            if duration is None:
                print(f"[Row {row_idx + 1}] Cannot read duration, skipping: {video_path}")
                total_errors += 1
                continue
            # Convert duration to HH:MM:SS (floor to whole seconds)
            dur_int = int(duration)
            h, rem = divmod(dur_int, 3600)
            m, s = divmod(rem, 60)
            timestamps = ["00:00:00", f"{h:02d}:{m:02d}:{s:02d}"]
        else:
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
        video_out_dir = output_base / video_stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        # Load progress for resume
        progress = load_progress(str(video_out_dir), run_params)

        if progress.get("video_complete"):
            print(f"[Row {row_idx + 1}] {video_stem}: already complete, skipping")
            completed_segs = progress.get("completed_segments", {})
            total_segments += len(completed_segs)
            total_frames_extracted += sum(
                s["num_extracted"] for s in completed_segs.values()
            )
            continue

        # Copy SRT file and compute relative path for metadata
        srt_relative_path = None
        if os.path.isfile(srt_path):
            srt_filename = Path(srt_path).name
            srt_dest = video_out_dir / srt_filename
            shutil.copy2(srt_path, srt_dest)
            # Relative path from a seg folder (one level down) to the SRT
            srt_relative_path = f"../{srt_filename}"
            print(f"  Copied SRT to {srt_dest}")

        num_timestamp_pairs = len(timestamps) // 2
        srt_info = f", SRT: {len(srt_entries)} entries" if srt_entries else ", SRT: not found"
        mode = "whole video" if whole_video else f"{num_timestamp_pairs} timestamp pair(s)"
        print(f"[Row {row_idx + 1}] {video_stem}: {mode}, {fps:.2f} fps{srt_info}")

        global_seg_counter = 1  # sequential across all segments of this video

        for i in range(num_timestamp_pairs):
            raw_start = timestamps[2 * i]
            raw_end = timestamps[2 * i + 1]

            try:
                start = normalize_timestamp(raw_start)
                end = normalize_timestamp(raw_end)
            except ValueError as e:
                print(f"  Pair {i + 1}: {e}, skipping")
                total_errors += 1
                continue

            # Resolve start frame number
            if srt_entries:
                start_frame, _ = lookup_frame_range(srt_entries, start, end)
            else:
                start_frame = round(timestamp_to_seconds(start) * fps)

            print(f"  Pair {i + 1}: {start} -> {end}, start_frame={start_frame}")

            global_seg_counter, extracted = extract_frames_for_segment(
                video_path=video_path,
                start_time=start,
                end_time=end,
                start_frame=start_frame,
                video_fps=fps,
                extract_fps=args.extract_fps,
                width=width,
                height=height,
                output_dir=str(video_out_dir),
                max_frames=args.max_frames,
                global_seg_counter=global_seg_counter,
                pair_index=i,
                srt_relative_path=srt_relative_path,
                progress=progress,
            )
            total_frames_extracted += extracted

        # Mark video as complete
        progress["video_complete"] = True
        save_progress(str(video_out_dir), progress)

        total_segments += global_seg_counter - 1
        print()

    print(f"Done. Created {total_segments} segment folder(s), "
          f"extracted {total_frames_extracted} frame(s), {total_errors} error(s).")


if __name__ == "__main__":
    main()
