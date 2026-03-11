#!/usr/bin/env python3
"""
Extract frames from a video for browsing and selection.

Saves frames as frame_XXXXXX.jpg (named by original video frame index)
along with extraction_info.json metadata for use by process_frames.py.

Usage:
    python scripts/extract_frames.py \
        --video_path /path/to/video.mp4 \
        --output_dir /path/to/extracted_frames \
        --frame_stride 3

    # Then browse the output directory to pick start/end frames,
    # and run process_frames.py with those frame numbers.
"""

import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for browsing and selection"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file (.mp4, .mov, .avi, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video path does not exist: {args.video_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video_path}")
        sys.exit(1)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate which frames to extract
    frame_indices = list(range(0, total_frames, args.frame_stride))
    if args.max_frames is not None and len(frame_indices) > args.max_frames:
        frame_indices = frame_indices[: args.max_frames]
    frame_indices_set = set(frame_indices)

    effective_fps = original_fps / args.frame_stride

    print(f"\nVideo: {args.video_path}")
    print(f"  - Original: {total_frames} frames @ {original_fps:.1f} FPS")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Stride: every {args.frame_stride} frame(s)")
    print(f"  - Extracting: {len(frame_indices)} frames")
    print(f"  - Effective FPS: {effective_fps:.1f}")
    print(f"  - Output: {args.output_dir}")

    # Extract frames, naming them by original video frame index
    # Also build index_mapping (sequential_idx -> original_frame_idx) for SAM3
    extracted_count = 0
    frame_idx = 0
    index_mapping = {}

    with tqdm(total=len(frame_indices), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_indices_set:
                # Save with original frame index in name for easy browsing
                output_path = os.path.join(
                    args.output_dir, f"frame_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                index_mapping[extracted_count] = frame_idx
                extracted_count += 1
                pbar.update(1)

                if args.max_frames is not None and extracted_count >= args.max_frames:
                    break

            frame_idx += 1

    cap.release()

    # Save extraction metadata
    extraction_info = {
        "video_path": os.path.abspath(args.video_path),
        "num_frames": extracted_count,
        "original_fps": original_fps,
        "effective_fps": effective_fps,
        "width": width,
        "height": height,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "frame_indices": frame_indices[:extracted_count],
        "index_mapping": {str(k): v for k, v in index_mapping.items()},
    }

    info_path = os.path.join(args.output_dir, "extraction_info.json")
    with open(info_path, "w") as f:
        json.dump(extraction_info, f, indent=2)

    print(f"\n{'='*50}")
    print("Extraction complete!")
    print(f"{'='*50}")
    print(f"  - Frames saved: {extracted_count}")
    print(f"  - Frame range: {frame_indices[0]} to {frame_indices[-1]}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Metadata: {info_path}")
    print(f"\nBrowse the frames in {args.output_dir}/ to pick your start and end frames.")
    print(f"Then run:")
    print(f"  python scripts/process_frames.py \\")
    print(f"    --frames_dir {args.output_dir} \\")
    print(f"    --output_dir <output_path> \\")
    print(f"    --start_frame <start> --end_frame <end> \\")
    print(f"    --text_prompt \"person\"")


if __name__ == "__main__":
    main()
