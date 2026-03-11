#!/usr/bin/env python3
"""
Run SAM3 on a selected range of previously extracted frames.

Works with frames extracted by extract_frames.py. You specify a start and end
frame (using the original video frame indices shown in filenames), and SAM3
processes only that range.

Usage:
    # First extract frames:
    python scripts/extract_frames.py --video_path video.mp4 --output_dir ./frames --frame_stride 3

    # Browse ./frames/ to pick start/end, then:
    python scripts/process_frames.py \
        --frames_dir ./frames \
        --output_dir ./output \
        --start_frame 30 --end_frame 150 \
        --text_prompt "person"
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Reuse utilities from process_video
from process_video import (
    create_color_map,
    create_output_video,
    create_prompt_subdir,
    overlay_masks_on_frame,
    save_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 on a selected range of extracted frames"
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory of extracted frames (output of extract_frames.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save SAM3 results",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        required=True,
        help="Start frame index (original video frame number from filename)",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        required=True,
        help="End frame index (original video frame number from filename, inclusive)",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="person",
        help="Text prompt for SAM3 segmentation (default: 'person')",
    )
    parser.add_argument(
        "--prompt_frame",
        type=int,
        default=None,
        help="Frame index (in extracted sequence) to add the prompt on (default: 0, i.e. first frame in range)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Frames per processing chunk (default: all at once)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2'). Default: all available",
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        default=False,
        help="Offload video/state to CPU (slower but less GPU memory)",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        default=True,
        help="Save binary masks as PNG",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=True,
        help="Save visualization images with masks overlaid",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Create output video from visualizations",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate frames directory
    if not os.path.isdir(args.frames_dir):
        print(f"Error: Frames directory does not exist: {args.frames_dir}")
        sys.exit(1)

    # Load extraction info if available
    info_path = os.path.join(args.frames_dir, "extraction_info.json")
    extraction_info = None
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            extraction_info = json.load(f)
        print(f"Loaded extraction info from {info_path}")
        print(f"  - Source video: {extraction_info.get('video_path', 'unknown')}")
        print(f"  - Total extracted frames: {extraction_info['num_frames']}")
        print(f"  - Frame stride: {extraction_info.get('frame_stride', 1)}")

    # Find all extracted frames and their original indices
    frame_files = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg")))
    if not frame_files:
        print(f"Error: No frame_*.jpg files found in {args.frames_dir}")
        sys.exit(1)

    # Parse frame indices from filenames
    import re

    available_frames = {}  # original_idx -> file_path
    for fp in frame_files:
        match = re.search(r"frame_(\d+)\.jpg", os.path.basename(fp))
        if match:
            idx = int(match.group(1))
            available_frames[idx] = fp

    all_indices = sorted(available_frames.keys())
    print(f"\nAvailable frames: {len(all_indices)} (range {all_indices[0]} to {all_indices[-1]})")

    # Filter to the selected range
    selected_indices = [i for i in all_indices if args.start_frame <= i <= args.end_frame]
    if not selected_indices:
        print(f"Error: No frames found in range [{args.start_frame}, {args.end_frame}]")
        print(f"Available range: [{all_indices[0]}, {all_indices[-1]}]")
        sys.exit(1)

    print(f"Selected frames: {len(selected_indices)} (range {selected_indices[0]} to {selected_indices[-1]})")

    # Create a temporary directory with sequentially numbered frames for SAM3
    # SAM3 expects frames named 000000.jpg, 000001.jpg, ...
    temp_dir = tempfile.mkdtemp(prefix="sam3_selected_")
    index_mapping = {}  # sequential_idx -> original_frame_idx

    print(f"\nPreparing frames for SAM3...")
    for seq_idx, orig_idx in enumerate(selected_indices):
        src = available_frames[orig_idx]
        dst = os.path.join(temp_dir, f"{seq_idx:06d}.jpg")
        os.symlink(os.path.abspath(src), dst)
        index_mapping[seq_idx] = orig_idx

    total_selected = len(selected_indices)
    prompt_frame = args.prompt_frame if args.prompt_frame is not None else 0

    # Create prompt-specific output directory
    output_dir = create_prompt_subdir(args.output_dir, args.text_prompt)
    os.makedirs(output_dir, exist_ok=True)

    # Setup GPUs
    if args.gpus is not None:
        gpus_to_use = [int(g) for g in args.gpus.split(",")]
    else:
        gpus_to_use = list(range(torch.cuda.device_count()))

    print(f"\n{'='*50}")
    print("SAM3 Processing")
    print(f"{'='*50}")
    print(f"  - Frame range: {selected_indices[0]} to {selected_indices[-1]} (original)")
    print(f"  - Frames to process: {total_selected}")
    print(f"  - Text prompt: '{args.text_prompt}'")
    print(f"  - Prompt frame: {prompt_frame} (sequential index)")
    print(f"  - GPUs: {gpus_to_use}")
    print(f"  - Output: {output_dir}")

    # Build SAM3 predictor
    print("\nBuilding SAM3 video predictor...")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    try:
        # Create output subdirectories
        if args.save_masks:
            os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        if args.save_visualizations:
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

        # Color map for objects
        colors = create_color_map(100)

        # Start SAM3 session on the temp directory of selected frames
        print("\nStarting SAM3 session...")
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=temp_dir,
                offload_video_to_cpu=args.offload_to_cpu,
                offload_state_to_cpu=args.offload_to_cpu,
            )
        )
        session_id = response["session_id"]

        # Add text prompt
        print(f"Adding text prompt '{args.text_prompt}' on frame {prompt_frame}...")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame,
                text=args.text_prompt,
            )
        )

        initial_objects = response.get("outputs", {})
        obj_ids_found = list(initial_objects.get("obj_ids", []))
        print(f"Found {len(obj_ids_found)} objects on prompt frame")

        # Propagate through selected frames
        all_outputs = {}
        total_detections = 0
        frames_with_detections = 0
        all_obj_ids_seen = set()

        chunk_size = args.chunk_size if args.chunk_size else total_selected

        chunks = []
        start = 0
        while start < total_selected:
            end = min(start + chunk_size, total_selected)
            chunks.append((start, end))
            start = end

        print(f"\nProcessing in {len(chunks)} chunk(s)...")

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = chunk_end - chunk_start
            print(f"\nChunk {chunk_idx + 1}/{len(chunks)}: sequential frames {chunk_start} to {chunk_end - 1}")

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
                desc=f"Processing chunk {chunk_idx + 1}",
            ):
                frame_idx = response["frame_index"]  # sequential index
                outputs = response["outputs"]
                all_outputs[frame_idx] = outputs

                out_binary_masks = outputs.get("out_binary_masks", None)
                out_obj_ids = outputs.get("out_obj_ids", None)

                # Map back to original frame index for saving
                save_frame_idx = index_mapping.get(frame_idx, frame_idx)

                has_detections = (
                    out_obj_ids is not None
                    and len(out_obj_ids) > 0
                    and out_binary_masks is not None
                    and len(out_binary_masks) > 0
                )

                if has_detections:
                    frames_with_detections += 1
                    total_detections += len(out_obj_ids)
                    for oid in out_obj_ids:
                        oid_val = int(oid.item()) if hasattr(oid, "item") else int(oid)
                        all_obj_ids_seen.add(oid_val)

                if has_detections:
                    masks = (
                        out_binary_masks.cpu().numpy()
                        if hasattr(out_binary_masks, "cpu")
                        else out_binary_masks
                    )

                    if args.save_masks:
                        for obj_id, mask in zip(out_obj_ids, masks):
                            save_mask(mask, obj_id, save_frame_idx, output_dir)

                    if args.save_visualizations:
                        frame_path = os.path.join(temp_dir, f"{frame_idx:06d}.jpg")
                        frame = np.array(Image.open(frame_path))
                        vis_frame = overlay_masks_on_frame(
                            frame, masks, out_obj_ids, colors
                        )
                        vis_path = os.path.join(
                            output_dir,
                            "visualizations",
                            f"frame_{save_frame_idx:06d}.jpg",
                        )
                        Image.fromarray(vis_frame).save(vis_path, quality=95)
                elif args.save_visualizations:
                    frame_path = os.path.join(temp_dir, f"{frame_idx:06d}.jpg")
                    frame = np.array(Image.open(frame_path))
                    vis_path = os.path.join(
                        output_dir,
                        "visualizations",
                        f"frame_{save_frame_idx:06d}.jpg",
                    )
                    Image.fromarray(frame).save(vis_path, quality=95)

        # Close session
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )

        # Print summary
        print(f"\n{'='*50}")
        print("Detection Summary:")
        print(f"{'='*50}")
        print(f"  - Frames processed: {len(all_outputs)}")
        print(f"  - Frames with detections: {frames_with_detections}/{len(all_outputs)}")
        print(f"  - Unique objects tracked: {len(all_obj_ids_seen)}")
        print(f"  - Object IDs: {sorted(all_obj_ids_seen) if all_obj_ids_seen else 'None'}")

        if frames_with_detections == 0:
            print(f"\n  WARNING: No objects detected!")
            print(f"  Try a different prompt or a different --prompt_frame")

        # Save metadata
        fps = extraction_info["effective_fps"] if extraction_info else 30.0
        metadata = {
            "source_video": extraction_info.get("video_path", "unknown") if extraction_info else "unknown",
            "frames_dir": os.path.abspath(args.frames_dir),
            "text_prompt": args.text_prompt,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "total_frames_processed": len(all_outputs),
            "fps": fps,
            "resolution": [
                extraction_info["width"] if extraction_info else 0,
                extraction_info["height"] if extraction_info else 0,
            ],
            "frames_with_detections": frames_with_detections,
            "unique_objects_tracked": len(all_obj_ids_seen),
            "object_ids": sorted(all_obj_ids_seen) if all_obj_ids_seen else [],
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create output video if requested
        if args.save_video:
            create_output_video(output_dir, fps=fps)

        print(f"\n{'='*50}")
        print("Processing complete!")
        print(f"{'='*50}")
        print(f"  - Output: {output_dir}")
        if args.save_masks:
            print(f"  - Masks: {os.path.join(output_dir, 'masks')}")
        if args.save_visualizations:
            print(f"  - Visualizations: {os.path.join(output_dir, 'visualizations')}")
        if args.save_video:
            print(f"  - Video: {os.path.join(output_dir, 'output_video.mp4')}")

    finally:
        print("\nShutting down predictor...")
        predictor.shutdown()

        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
