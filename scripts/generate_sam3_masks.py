#!/usr/bin/env python3
"""
Generate SAM3 segmentation masks for zipped video segments.

Takes a zip of pre-extracted frame segments (from extract_segments_frames.py),
runs SAM3 text-prompted segmentation on each segment, and produces a new zip
with binary masks added.

Input zip structure:
    vid1/
        vid1.SRT
        seg1/
            frame_000300.jpg ... (non-contiguous, original video frame numbers)
            metadata.json
        seg2/ ...
    vid2/ ...

Output adds per segment:
    seg1/sam3_masks/
        masks/obj_0/frame_000300.png ...
        metadata.json

Usage:
    python scripts/generate_sam3_masks.py \
        --input-zip /path/to/segments.zip \
        --output-zip /path/to/segments_with_masks.zip \
        --text-prompt "zebra"
"""

import argparse
import gc
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAM3 masks for zipped video segments"
    )
    parser.add_argument("--input-zip", type=str, required=True,
                        help="Path to input zip with extracted frame segments")
    parser.add_argument("--output-zip", type=str, required=True,
                        help="Path to output zip (input + masks)")
    parser.add_argument("--text-prompt", type=str, required=True,
                        help="SAM3 text prompt (e.g., 'zebra', 'animal')")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="SAM3 chunk size for processing (default: 500)")
    parser.add_argument("--prompt-frame", type=int, default=0,
                        help="Frame index (in extracted sequence) to add prompt on (default: 0)")
    parser.add_argument("--save-visualizations", action="store_true", default=False,
                        help="Also save visualization frames with mask overlays")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (default: all available)")
    return parser.parse_args()


def discover_segments(work_dir: str):
    """
    Walk the work directory and find all segment folders (those with metadata.json).
    Returns list of (video_dir, seg_dir, metadata) tuples sorted by path.
    """
    segments = []
    for root, dirs, files in os.walk(work_dir):
        if "metadata.json" in files:
            meta_path = os.path.join(root, "metadata.json")
            # Skip SAM3 mask metadata or VGGT metadata
            if "sam3_masks" in root or "vggt_results" in root:
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                # Must have frame_numbers to be a valid segment
                if "frame_numbers" in meta:
                    video_dir = os.path.dirname(root)
                    segments.append((video_dir, root, meta))
            except (json.JSONDecodeError, KeyError):
                continue

    segments.sort(key=lambda x: x[1])
    return segments


def load_progress(work_dir: str, params: dict) -> dict:
    """Load or create sam3_progress.json."""
    progress_path = os.path.join(work_dir, "sam3_progress.json")
    fresh = {"params": params, "completed_segments": [], "timestamp": None}

    if not os.path.isfile(progress_path):
        return fresh

    with open(progress_path) as f:
        progress = json.load(f)

    if progress.get("params") != params:
        print("  SAM3 parameters changed, restarting mask generation")
        return fresh

    return progress


def save_progress(work_dir: str, progress: dict):
    """Save sam3_progress.json atomically."""
    progress_path = os.path.join(work_dir, "sam3_progress.json")
    progress["timestamp"] = datetime.now().isoformat()
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, progress_path)


def is_segment_masks_complete(seg_dir: str):
    """Check if SAM3 masks already exist and look complete for a segment."""
    masks_dir = os.path.join(seg_dir, "sam3_masks")
    if not os.path.isdir(masks_dir):
        return False
    meta_path = os.path.join(masks_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return False
    # Check at least one obj directory with masks
    mask_root = os.path.join(masks_dir, "masks")
    if not os.path.isdir(mask_root):
        return False
    obj_dirs = glob.glob(os.path.join(mask_root, "obj_*"))
    return len(obj_dirs) > 0


def create_sequential_symlinks(seg_dir: str, frame_numbers: list, temp_dir: str):
    """
    Create a temp directory with sequential symlinks for SAM3.
    SAM3 expects 000000.jpg, 000001.jpg, etc.

    Returns:
        index_mapping: dict mapping sequential index → original frame number
    """
    # Get sorted JPG files from segment
    frame_files = {}
    for fn in frame_numbers:
        path = os.path.join(seg_dir, f"frame_{fn:06d}.jpg")
        if os.path.isfile(path):
            frame_files[fn] = path

    index_mapping = {}
    for seq_idx, fn in enumerate(sorted(frame_files.keys())):
        src = frame_files[fn]
        dst = os.path.join(temp_dir, f"{seq_idx:06d}.jpg")
        os.symlink(os.path.abspath(src), dst)
        index_mapping[seq_idx] = fn

    return index_mapping


def rename_masks_to_original(masks_dir: str, index_mapping: dict):
    """
    Rename mask files from sequential (frame_000000.png) to original
    frame numbers (frame_000300.png) using the index mapping.
    """
    for obj_dir in glob.glob(os.path.join(masks_dir, "masks", "obj_*")):
        for mask_file in glob.glob(os.path.join(obj_dir, "frame_*.png")):
            basename = os.path.basename(mask_file)
            match = re.match(r"frame_(\d+)\.png", basename)
            if match:
                seq_idx = int(match.group(1))
                if seq_idx in index_mapping:
                    original_fn = index_mapping[seq_idx]
                    new_name = f"frame_{original_fn:06d}.png"
                    new_path = os.path.join(obj_dir, new_name)
                    if mask_file != new_path:
                        os.rename(mask_file, new_path)


def save_mask(mask, obj_id, frame_idx, output_dir):
    """Save a single binary mask as PNG."""
    obj_id_val = int(obj_id.item()) if hasattr(obj_id, "item") else int(obj_id)
    mask_dir = os.path.join(output_dir, "masks", f"obj_{obj_id_val}")
    os.makedirs(mask_dir, exist_ok=True)

    if mask.ndim > 2:
        mask = mask.squeeze()

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_path = os.path.join(mask_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(mask_path, mask_uint8)


def process_segment_sam3(predictor, seg_dir: str, metadata: dict,
                         text_prompt: str, chunk_size: int, prompt_frame: int,
                         save_visualizations: bool):
    """
    Run SAM3 on a single segment's frames and save masks.

    Creates seg_dir/sam3_masks/ with the standard mask format.
    """
    frame_numbers = metadata["frame_numbers"]
    seg_name = os.path.basename(seg_dir)

    # Create temp dir with sequential symlinks
    temp_dir = tempfile.mkdtemp(prefix=f"sam3_{seg_name}_")

    try:
        index_mapping = create_sequential_symlinks(seg_dir, frame_numbers, temp_dir)
        num_frames = len(index_mapping)

        if num_frames == 0:
            print(f"    No frames found in {seg_dir}, skipping")
            return False

        print(f"    Created {num_frames} sequential symlinks for SAM3")

        # Output directory for masks
        masks_output = os.path.join(seg_dir, "sam3_masks")
        os.makedirs(os.path.join(masks_output, "masks"), exist_ok=True)

        # Start SAM3 session on the temp directory
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=temp_dir,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )
        )
        session_id = response["session_id"]

        # Add text prompt
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame,
                text=text_prompt,
            )
        )
        obj_ids_found = list(response.get("outputs", {}).get("obj_ids", []))
        print(f"    Found {len(obj_ids_found)} objects on prompt frame")

        # Process in chunks
        frames_with_detections = 0
        all_obj_ids_seen = set()

        chunks = []
        start = 0
        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            chunks.append((start, end))
            start = end

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = chunk_end - chunk_start
            use_start_frame = chunk_start if chunk_idx > 0 else None

            for response in tqdm(
                predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_index=use_start_frame,
                        max_frame_num_to_track=chunk_frames,
                    )
                ),
                total=chunk_frames,
                desc=f"    Chunk {chunk_idx + 1}/{len(chunks)}",
            ):
                frame_idx = response["frame_index"]
                outputs = response["outputs"]

                out_binary_masks = outputs.get("out_binary_masks", None)
                out_obj_ids = outputs.get("out_obj_ids", None)

                has_detections = (
                    out_obj_ids is not None and len(out_obj_ids) > 0
                    and out_binary_masks is not None and len(out_binary_masks) > 0
                )

                if has_detections:
                    frames_with_detections += 1
                    masks = out_binary_masks.cpu().numpy() if hasattr(out_binary_masks, "cpu") else out_binary_masks

                    for obj_id, mask in zip(out_obj_ids, masks):
                        oid_val = int(obj_id.item()) if hasattr(obj_id, "item") else int(obj_id)
                        all_obj_ids_seen.add(oid_val)
                        # Save with sequential index — will rename later
                        save_mask(mask, obj_id, frame_idx, masks_output)

        # Close session
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )

        # Rename masks from sequential to original frame numbers
        print(f"    Renaming masks to original frame numbers...")
        rename_masks_to_original(masks_output, index_mapping)

        # Save SAM3 metadata
        sam3_metadata = {
            "text_prompt": text_prompt,
            "total_frames": num_frames,
            "fps": metadata.get("extract_fps", metadata.get("video_fps")),
            "resolution": None,  # will be filled from first frame if needed
            "frames_with_detections": frames_with_detections,
            "unique_objects_tracked": len(all_obj_ids_seen),
            "object_ids": sorted(all_obj_ids_seen),
            "frame_numbers": sorted(index_mapping.values()),
            "extracted_on": datetime.now().isoformat(),
        }

        # Get resolution from first frame
        first_frame_path = os.path.join(seg_dir, f"frame_{frame_numbers[0]:06d}.jpg")
        if os.path.isfile(first_frame_path):
            img = cv2.imread(first_frame_path)
            if img is not None:
                h, w = img.shape[:2]
                sam3_metadata["resolution"] = [w, h]

        meta_path = os.path.join(masks_output, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(sam3_metadata, f, indent=2)

        print(f"    Detections: {frames_with_detections}/{num_frames} frames, "
              f"{len(all_obj_ids_seen)} unique objects")

        return True

    finally:
        # Clean up temp symlink directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.input_zip):
        print(f"Input zip not found: {args.input_zip}")
        sys.exit(1)

    # Unzip to temp directory
    work_dir = tempfile.mkdtemp(prefix="sam3_work_")
    print(f"Extracting {args.input_zip} to {work_dir}...")

    with zipfile.ZipFile(args.input_zip, "r") as zf:
        zf.extractall(work_dir)

    print(f"Extracted to {work_dir}")

    # Discover all segments
    segments = discover_segments(work_dir)
    print(f"Found {len(segments)} segment(s)\n")

    if not segments:
        print("No segments found in zip. Expected metadata.json with frame_numbers.")
        shutil.rmtree(work_dir)
        sys.exit(1)

    # Progress tracking
    run_params = {"text_prompt": args.text_prompt, "chunk_size": args.chunk_size}
    progress = load_progress(work_dir, run_params)
    completed = set(progress.get("completed_segments", []))

    # Setup GPUs
    if args.gpus is not None:
        gpus_to_use = [int(g) for g in args.gpus.split(",")]
    else:
        gpus_to_use = list(range(torch.cuda.device_count()))

    print(f"Using GPUs: {gpus_to_use}")

    # Build SAM3 predictor
    print("Building SAM3 video predictor...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    total_processed = 0
    total_skipped = 0

    try:
        for video_dir, seg_dir, metadata in segments:
            seg_name = os.path.basename(seg_dir)
            video_name = os.path.basename(video_dir)
            seg_id = f"{video_name}/{seg_name}"

            # Check if already complete
            if seg_id in completed and is_segment_masks_complete(seg_dir):
                print(f"[{seg_id}] SKIP (already complete)")
                total_skipped += 1
                continue

            # Clean up incomplete masks if present
            masks_dir = os.path.join(seg_dir, "sam3_masks")
            if os.path.isdir(masks_dir):
                shutil.rmtree(masks_dir)

            pair_label = metadata.get("pair_seg_label", seg_name)
            num_frames = len(metadata.get("frame_numbers", []))
            print(f"[{seg_id}] Processing {num_frames} frames ({pair_label})...")

            ok = process_segment_sam3(
                predictor=predictor,
                seg_dir=seg_dir,
                metadata=metadata,
                text_prompt=args.text_prompt,
                chunk_size=args.chunk_size,
                prompt_frame=args.prompt_frame,
                save_visualizations=args.save_visualizations,
            )

            if ok:
                total_processed += 1
                progress["completed_segments"].append(seg_id)
                save_progress(work_dir, progress)

            # Clear GPU memory between segments
            torch.cuda.empty_cache()
            gc.collect()

    finally:
        print("\nShutting down SAM3 predictor...")
        predictor.shutdown()
        torch.cuda.empty_cache()
        gc.collect()

    # Create output zip
    print(f"\nCreating output zip: {args.output_zip}...")
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, work_dir)
                zf.write(file_path, arcname)

    # Clean up work directory
    print(f"Cleaning up {work_dir}...")
    shutil.rmtree(work_dir)

    print(f"\nDone. Processed {total_processed} segment(s), "
          f"skipped {total_skipped} (already complete).")
    print(f"Output: {args.output_zip}")


if __name__ == "__main__":
    main()
