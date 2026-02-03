#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 Video Processing Script

Process videos (including long ones in chunks) with SAM3 and save segmentation results.

Usage:
    python scripts/process_video.py \
        --video_path /path/to/video.mp4 \
        --output_dir /path/to/output \
        --text_prompt "person" \
        --chunk_size 500

For long videos, the script processes in chunks to manage memory efficiently.

Examples:
    # Process first 400 frames only
    python scripts/process_video.py --video_path video.mp4 --output_dir out --text_prompt "lion" --max_frames 400

    # Process every 3rd frame (reduce frame rate by 3x)
    python scripts/process_video.py --video_path video.mp4 --output_dir out --text_prompt "lion" --frame_stride 3

    # Combine both: every 2nd frame, max 500 frames
    python scripts/process_video.py --video_path video.mp4 --output_dir out --text_prompt "lion" --frame_stride 2 --max_frames 500

    # Track multiple object types
    python scripts/process_video.py --video_path video.mp4 --output_dir out --text_prompt "lion" "jackal" "buffalo"
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process video with SAM3 and save segmentation results"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file (.mp4) or JPEG frame folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output results",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        nargs="+",
        default=["person"],
        help="Text prompts describing objects to segment (e.g., 'lion' 'jackal' 'buffalo')",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of frames to process per chunk. If None, processes entire video at once. "
        "Use smaller values (e.g., 200-500) for long videos to manage memory.",
    )
    parser.add_argument(
        "--prompt_frame",
        type=int,
        default=0,
        help="Frame index to add the text prompt on (default: 0)",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        default=True,
        help="Save binary masks as PNG files",
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
        help="Save output as a video file with masks overlaid",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices to use (e.g., '0,1,2'). Default: all available",
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        default=False,
        help="Offload video frames and state to CPU (slower but uses less GPU memory)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process. If None, processes all frames.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames). "
        "Use 2 to halve frame rate, 3 to reduce by 3x, etc.",
    )
    return parser.parse_args()


def extract_frames_to_folder(
    video_path: str,
    output_folder: str,
    frame_stride: int = 1,
    max_frames: int = None,
):
    """
    Extract frames from video to a JPEG folder with optional stride and frame limit.

    Args:
        video_path: Path to input video file
        output_folder: Path to output folder for JPEG frames
        frame_stride: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract (None = no limit)

    Returns:
        dict with extraction info: num_frames, original_fps, effective_fps, frame_indices
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate which frames to extract
    frame_indices = list(range(0, total_frames, frame_stride))
    if max_frames is not None and len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]

    print(f"\nExtracting frames from video...")
    print(f"  - Original: {total_frames} frames @ {original_fps:.1f} FPS")
    print(f"  - Stride: every {frame_stride} frame(s)")
    print(f"  - Extracting: {len(frame_indices)} frames")

    effective_fps = original_fps / frame_stride

    extracted_count = 0
    frame_idx = 0

    # Map from extracted index to original frame index
    index_mapping = {}

    with tqdm(total=len(frame_indices), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_indices:
                # Save frame with sequential numbering (SAM3 expects this)
                output_path = os.path.join(output_folder, f"{extracted_count:06d}.jpg")
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                index_mapping[extracted_count] = frame_idx
                extracted_count += 1
                pbar.update(1)

                if max_frames is not None and extracted_count >= max_frames:
                    break

            frame_idx += 1

    cap.release()

    return {
        "num_frames": extracted_count,
        "original_fps": original_fps,
        "effective_fps": effective_fps,
        "width": width,
        "height": height,
        "frame_indices": frame_indices[:extracted_count],
        "index_mapping": index_mapping,
    }


def get_video_info(video_path: str):
    """Get video information (frame count, fps, dimensions)."""
    if video_path.endswith((".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV", ".webm", ".WEBM")):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}
    else:
        # JPEG folder
        import glob
        frames = glob.glob(os.path.join(video_path, "*.jpg"))
        if not frames:
            frames = glob.glob(os.path.join(video_path, "*.png"))
        frame_count = len(frames)
        if frame_count > 0:
            img = Image.open(frames[0])
            width, height = img.size
        else:
            width, height = 0, 0
        return {"frame_count": frame_count, "fps": 30, "width": width, "height": height}


def load_video_frames(video_path: str):
    """Load video frames for visualization."""
    video_extensions = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV", ".webm", ".WEBM")
    if video_path.endswith(video_extensions):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    else:
        import glob
        frame_paths = glob.glob(os.path.join(video_path, "*.jpg"))
        if not frame_paths:
            frame_paths = glob.glob(os.path.join(video_path, "*.png"))
        try:
            frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            frame_paths.sort()
        return frame_paths


def create_color_map(num_objects: int):
    """Create distinct colors for different object IDs."""
    np.random.seed(42)
    colors = {}
    for i in range(num_objects + 10):  # Extra colors for safety
        colors[i] = tuple(np.random.randint(50, 255, 3).tolist())
    return colors


def overlay_masks_on_frame(frame, masks, obj_ids, colors, alpha=0.5):
    """Overlay segmentation masks on a frame."""
    if isinstance(frame, str):
        frame = np.array(Image.open(frame))

    overlay = frame.copy()

    for obj_id, mask in zip(obj_ids, masks):
        if obj_id not in colors:
            colors[obj_id] = tuple(np.random.randint(50, 255, 3).tolist())
        color = colors[obj_id]

        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Create colored mask
        mask_bool = mask > 0
        overlay[mask_bool] = (
            np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)
        ).astype(np.uint8)

    return overlay


def save_mask(mask, obj_id, frame_idx, output_dir):
    """Save a single mask as PNG."""
    # Convert obj_id if it's a tensor
    obj_id_val = int(obj_id.item()) if hasattr(obj_id, 'item') else int(obj_id)
    mask_dir = os.path.join(output_dir, "masks", f"obj_{obj_id_val}")
    os.makedirs(mask_dir, exist_ok=True)

    if mask.ndim > 2:
        mask = mask.squeeze()

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_path = os.path.join(mask_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(mask_path, mask_uint8)


def process_video_in_chunks(
    predictor,
    video_path: str,
    output_dir: str,
    text_prompt: str,
    chunk_size: int = None,
    prompt_frame: int = 0,
    save_masks: bool = True,
    save_visualizations: bool = True,
    offload_to_cpu: bool = False,
    index_mapping: dict = None,
):
    """
    Process video in chunks and save results.

    For long videos, this processes the video in manageable chunks to avoid
    running out of GPU memory.

    Args:
        index_mapping: Optional dict mapping extracted frame indices to original frame indices.
                       Used when frames were extracted with stride/limit.
    """
    # Get video info
    video_info = get_video_info(video_path)
    total_frames = video_info["frame_count"]
    fps = video_info["fps"]

    # Normalize text_prompt to a list
    if isinstance(text_prompt, str):
        text_prompts = [text_prompt]
    else:
        text_prompts = list(text_prompt)

    print(f"\nVideo Info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  - Text prompts: {text_prompts}")

    if chunk_size is None:
        chunk_size = total_frames
        print(f"  - Processing: entire video at once")
    else:
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        print(f"  - Chunk size: {chunk_size} frames")
        print(f"  - Number of chunks: {num_chunks}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_masks:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    if save_visualizations:
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    # Load frames for visualization
    print("\nLoading video frames for visualization...")
    video_frames = load_video_frames(video_path)

    # Color map for objects
    colors = create_color_map(100)

    # Start session
    print("\nStarting SAM3 session...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
            offload_video_to_cpu=offload_to_cpu,
            offload_state_to_cpu=offload_to_cpu,
        )
    )
    session_id = response["session_id"]

    # Add text prompts for each object type
    obj_ids_found = []
    for prompt in text_prompts:
        print(f"Adding text prompt '{prompt}' on frame {prompt_frame}...")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame,
                text=prompt,
            )
        )

        initial_objects = response.get("outputs", {})
        prompt_obj_ids = list(initial_objects.get("obj_ids", []))
        obj_ids_found.extend(prompt_obj_ids)
        print(f"  Found {len(prompt_obj_ids)} objects for '{prompt}'")

    # Process in chunks
    all_outputs = {}
    total_detections = 0
    frames_with_detections = 0
    all_obj_ids_seen = set()

    # Calculate chunks
    chunks = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))
        start = end

    print(f"\nProcessing video in {len(chunks)} chunk(s)...")

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_frames = chunk_end - chunk_start
        print(f"\nChunk {chunk_idx + 1}/{len(chunks)}: frames {chunk_start} to {chunk_end - 1}")

        # Propagate for this chunk
        for response in tqdm(
            predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    start_frame_index=chunk_start if chunk_idx > 0 else None,
                    max_frame_num_to_track=chunk_frames,
                )
            ),
            total=chunk_frames,
            desc=f"Processing chunk {chunk_idx + 1}",
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]
            all_outputs[frame_idx] = outputs

            # Get masks and object IDs (correct keys from SAM3 output)
            out_binary_masks = outputs.get("out_binary_masks", None)
            out_obj_ids = outputs.get("out_obj_ids", None)

            # Get the original frame index for saving (if using stride/limit)
            save_frame_idx = index_mapping.get(frame_idx, frame_idx) if index_mapping else frame_idx

            # Track detections
            has_detections = (out_obj_ids is not None and len(out_obj_ids) > 0 and
                              out_binary_masks is not None and len(out_binary_masks) > 0)

            if has_detections:
                frames_with_detections += 1
                total_detections += len(out_obj_ids)
                for oid in out_obj_ids:
                    oid_val = int(oid.item()) if hasattr(oid, 'item') else int(oid)
                    all_obj_ids_seen.add(oid_val)

            if has_detections:
                # Masks are already binary from SAM3
                masks = out_binary_masks.cpu().numpy() if hasattr(out_binary_masks, 'cpu') else out_binary_masks

                # Save individual masks
                if save_masks:
                    for obj_id, mask in zip(out_obj_ids, masks):
                        save_mask(mask, obj_id, save_frame_idx, output_dir)

                # Save visualization with masks
                if save_visualizations:
                    frame = video_frames[frame_idx]
                    vis_frame = overlay_masks_on_frame(frame, masks, out_obj_ids, colors)
                    vis_path = os.path.join(output_dir, "visualizations", f"frame_{save_frame_idx:06d}.jpg")
                    Image.fromarray(vis_frame).save(vis_path, quality=95)
            elif save_visualizations:
                # Save frame without masks (no objects detected)
                frame = video_frames[frame_idx]
                if isinstance(frame, str):
                    frame = np.array(Image.open(frame))
                vis_path = os.path.join(output_dir, "visualizations", f"frame_{save_frame_idx:06d}.jpg")
                Image.fromarray(frame).save(vis_path, quality=95)

    # Print detection summary
    print(f"\n{'='*50}")
    print("Detection Summary:")
    print(f"{'='*50}")
    print(f"  - Frames with detections: {frames_with_detections}/{len(all_outputs)}")
    print(f"  - Unique objects tracked: {len(all_obj_ids_seen)}")
    print(f"  - Object IDs: {sorted(all_obj_ids_seen) if all_obj_ids_seen else 'None'}")
    if frames_with_detections == 0:
        print(f"\n  ⚠️  NO OBJECTS DETECTED!")
        print(f"  Try different prompts or more generic terms like 'animal' or 'wildlife'")
        print(f"  Or try a different --prompt_frame (e.g., 50, 100)")

    # Save metadata
    metadata = {
        "video_path": video_path,
        "text_prompts": text_prompts,
        "total_frames": total_frames,
        "fps": fps,
        "resolution": [video_info["width"], video_info["height"]],
        "chunk_size": chunk_size if chunk_size != total_frames else None,
        "objects_found": obj_ids_found,
        "frames_processed": len(all_outputs),
        "frames_with_detections": frames_with_detections,
        "unique_objects_tracked": len(all_obj_ids_seen),
        "object_ids": sorted(all_obj_ids_seen) if all_obj_ids_seen else [],
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")

    # Close session
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    return all_outputs, metadata


def create_output_video(output_dir: str, fps: float = 30.0):
    """Create a video from visualization frames."""
    vis_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(vis_dir):
        print("No visualizations found to create video")
        return

    import glob
    frames = sorted(glob.glob(os.path.join(vis_dir, "frame_*.jpg")))
    if not frames:
        print("No visualization frames found")
        return

    # Get frame size from first frame
    first_frame = cv2.imread(frames[0])
    height, width = first_frame.shape[:2]

    # Create video writer
    video_path = os.path.join(output_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"\nCreating output video: {video_path}")
    for frame_path in tqdm(frames, desc="Writing video"):
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {video_path}")


def main():
    args = parse_args()

    # Validate input
    if not os.path.exists(args.video_path):
        print(f"Error: Video path does not exist: {args.video_path}")
        sys.exit(1)

    # Setup GPUs
    if args.gpus is not None:
        gpus_to_use = [int(g) for g in args.gpus.split(",")]
    else:
        gpus_to_use = list(range(torch.cuda.device_count()))

    print(f"Using GPUs: {gpus_to_use}")

    # Check if we need to extract frames (stride > 1 or max_frames specified)
    need_extraction = args.frame_stride > 1 or args.max_frames is not None
    temp_dir = None
    video_path_to_use = args.video_path
    index_mapping = None
    effective_fps = None

    video_extensions = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV", ".webm", ".WEBM")
    if need_extraction and args.video_path.endswith(video_extensions):
        # Create temp directory for extracted frames
        temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        print(f"\nExtracting frames to temporary folder: {temp_dir}")

        extraction_info = extract_frames_to_folder(
            video_path=args.video_path,
            output_folder=temp_dir,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )

        video_path_to_use = temp_dir
        index_mapping = extraction_info["index_mapping"]
        effective_fps = extraction_info["effective_fps"]

        print(f"  - Effective FPS: {effective_fps:.1f}")

    # Build predictor
    print("\nBuilding SAM3 video predictor...")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    try:
        # Process video
        outputs, metadata = process_video_in_chunks(
            predictor=predictor,
            video_path=video_path_to_use,
            output_dir=args.output_dir,
            text_prompt=args.text_prompt,
            chunk_size=args.chunk_size,
            prompt_frame=args.prompt_frame,
            save_masks=args.save_masks,
            save_visualizations=args.save_visualizations,
            offload_to_cpu=args.offload_to_cpu,
            index_mapping=index_mapping,
        )

        # Update metadata with extraction info
        if need_extraction:
            metadata["frame_stride"] = args.frame_stride
            metadata["max_frames"] = args.max_frames
            metadata["effective_fps"] = effective_fps
            metadata["original_video_path"] = args.video_path

            # Re-save metadata
            metadata_path = os.path.join(args.output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Create output video if requested
        output_fps = effective_fps if effective_fps else metadata["fps"]
        if args.save_video:
            create_output_video(args.output_dir, fps=output_fps)

        print(f"\n{'='*50}")
        print("Processing complete!")
        print(f"{'='*50}")
        print(f"Output directory: {args.output_dir}")
        print(f"Frames processed: {metadata['frames_processed']}")
        if args.frame_stride > 1:
            print(f"Frame stride: {args.frame_stride} (every {args.frame_stride} frames)")
        if args.max_frames:
            print(f"Max frames limit: {args.max_frames}")
        print(f"Objects tracked: {len(metadata['objects_found'])}")
        if args.save_masks:
            print(f"Masks saved to: {os.path.join(args.output_dir, 'masks')}")
        if args.save_visualizations:
            print(f"Visualizations saved to: {os.path.join(args.output_dir, 'visualizations')}")
        if args.save_video:
            print(f"Output video: {os.path.join(args.output_dir, 'output_video.mp4')}")

    finally:
        # Clean up
        print("\nShutting down predictor...")
        predictor.shutdown()

        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            print(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
