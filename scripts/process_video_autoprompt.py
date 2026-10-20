#!/usr/bin/env python3
"""
SAM3 video processing with automatic prompt-frame selection and adjustable tracking thresholds.

Why this exists
    The text prompt applies to every frame, but the frame it is added on ("prompt frame") is special:
    detections there only need a score of 0.5 and propagation starts from it. An animal that is not
    visible on frame 0 is picked up later only as a *new* object, which needs a score of 0.7 and must
    survive the hot-start check (a new track is dropped if it goes unmatched for 8 of its first 15
    frames). Faint animals, e.g. at night, often clear the first bar but not the second.

What this script does
    1. SCAN: opens a session, and for every N-th frame calls add_prompt with the text prompt. Each call
       re-runs detection on that single frame (the session state is reset by every semantic prompt),
       so the scan costs one detector pass per sampled frame. Scores per frame are saved to
       prompt_scan.json in the output folder. This is also a "presence" trace of the animal.
    2. PICK: the frame with the strongest detection becomes the prompt frame (override with
       --prompt_frame; disable the scan with --scan_stride 0).
    3. RUN: one propagation in both directions from that frame, exactly as process_video.py.
    4. Optional threshold overrides (--new_det_thresh, --score_threshold_detection, hot-start
       parameters) applied to the loaded model; useful to trade false positives for recall on hard clips.
       They take effect on the main GPU process; use a single GPU (--gpus 0) when changing them.

Usage
    python scripts/process_video_autoprompt.py --video_path clip.AVI --output_dir out --text_prompt deer \
        --max_frames 300 --scan_stride 10 --save_video
    # relaxed tracking for faint night-time animals:
    python scripts/process_video_autoprompt.py ... --new_det_thresh 0.5 --hotstart_unmatch_thresh 12
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_video import (  # noqa: E402
    create_output_video,
    create_prompt_subdir,
    detect_resume_point,
    extract_frames_to_folder,
    get_video_info,
    process_video_in_chunks,
)

THRESHOLD_ARGS = (
    "score_threshold_detection",  # detection score needed on the prompt frame (default 0.5)
    "new_det_thresh",             # score needed to start a NEW object during propagation (default 0.7)
    "hotstart_delay",             # frames a new track is held back before it is trusted (default 15)
    "hotstart_unmatch_thresh",    # drop a new track unmatched for this many frames within the delay (default 8)
    "hotstart_dup_thresh",        # drop a new track overlapping an older one for this many frames (default 8)
)


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 video processing with automatic prompt frame")
    p.add_argument("--video_path", required=True, help="video file (.mp4/.avi/.mov/...) or JPEG frame folder")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--text_prompt", default="animal")
    p.add_argument("--prompt_frame", type=int, default=None,
                   help="use this frame (index in the processed sequence) and skip the scan")
    p.add_argument("--scan_stride", type=int, default=10,
                   help="scan every N-th frame for the best prompt frame (0 = no scan, use frame 0)")
    p.add_argument("--max_scan_frames", type=int, default=60,
                   help="cap on scanned frames; the sample is thinned evenly above this")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--save_video", action="store_true", default=False)
    p.add_argument("--gpus", type=str, default=None, help="comma-separated GPU ids (default: all)")
    p.add_argument("--offload_to_cpu", action="store_true", default=False)
    for name in THRESHOLD_ARGS:
        p.add_argument(f"--{name}", type=float if "thresh" in name and "hotstart" not in name else int, default=None,
                       help=f"override the model's {name}")
    return p.parse_args()


def apply_threshold_overrides(predictor, args):
    model = getattr(predictor, "model", None)
    if model is None:
        return {}
    applied = {}
    for name in THRESHOLD_ARGS:
        value = getattr(args, name)
        if value is None:
            continue
        old = getattr(model, name, None)
        setattr(model, name, value)
        applied[name] = {"old": old, "new": value}
        print(f"  - override {name}: {old} -> {value}")
    if applied and getattr(predictor, "world_size", 1) > 1:
        print("  WARNING: threshold overrides only reach the main GPU process; run with --gpus 0 for consistency")
    if "hotstart_delay" in applied or "hotstart_unmatch_thresh" in applied or "hotstart_dup_thresh" in applied:
        d = getattr(model, "hotstart_delay", 0)
        if getattr(model, "hotstart_unmatch_thresh", 0) > d or getattr(model, "hotstart_dup_thresh", 0) > d:
            sys.exit("hotstart_unmatch_thresh and hotstart_dup_thresh must not exceed hotstart_delay")
    return applied


def scan_prompt_frames(predictor, video_path, text_prompt, scan_stride, max_scan, offload_to_cpu, index_mapping):
    """Run the detector with the prompt on sampled frames; return one row per frame with its scores."""
    num_frames = get_video_info(video_path)["frame_count"]
    frames = list(range(0, num_frames, max(scan_stride, 1)))
    if max_scan and len(frames) > max_scan:
        step = len(frames) / max_scan
        frames = [frames[int(i * step)] for i in range(max_scan)]
    print(f"\nScanning {len(frames)} frames (every {scan_stride}) for the best prompt frame...")
    response = predictor.handle_request(request=dict(
        type="start_session", resource_path=video_path,
        offload_video_to_cpu=offload_to_cpu, offload_state_to_cpu=offload_to_cpu))
    session_id = response["session_id"]
    rows = []
    try:
        for k in frames:
            out = predictor.handle_request(request=dict(
                type="add_prompt", session_id=session_id, frame_index=k, text=text_prompt)).get("outputs", {})
            probs = [float(x) for x in out.get("out_probs", [])]
            rows.append(dict(frame=int(k), original_frame=int(index_mapping.get(k, k)) if index_mapping else int(k),
                             n_objects=len(probs), max_score=max(probs) if probs else 0.0,
                             sum_score=float(sum(probs))))
    finally:
        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return rows


def choose_prompt_frame(rows):
    best = max(rows, key=lambda r: (r["max_score"], r["n_objects"]))
    return best


def print_scan(rows, best):
    print(f"  {'frame':>7} {'orig':>7} {'objects':>8} {'max score':>10}")
    for r in rows:
        mark = "  <- prompt frame" if r is best else ""
        print(f"  {r['frame']:>7} {r['original_frame']:>7} {r['n_objects']:>8} {r['max_score']:>10.3f}{mark}")
    seen = [r for r in rows if r["n_objects"] > 0]
    if not seen:
        print("  no detections on any scanned frame; propagation will still run from frame 0")
    else:
        print(f"  detections on {len(seen)}/{len(rows)} scanned frames "
              f"(first at frame {seen[0]['frame']}, last at frame {seen[-1]['frame']})")


def main():
    args = parse_args()
    if not os.path.exists(args.video_path):
        sys.exit(f"video not found: {args.video_path}")
    output_dir = create_prompt_subdir(args.output_dir, args.text_prompt)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    gpus_to_use = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))

    # frame extraction when a stride or cap is requested (same as process_video.py)
    video_extensions = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV", ".webm", ".WEBM")
    need_extraction = (args.frame_stride > 1 or args.max_frames is not None) and args.video_path.endswith(video_extensions)
    temp_dir, index_mapping, effective_fps = None, None, None
    video_path_to_use = args.video_path
    if need_extraction:
        temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        info = extract_frames_to_folder(args.video_path, temp_dir, args.frame_stride, args.max_frames)
        video_path_to_use, index_mapping, effective_fps = temp_dir, info["index_mapping"], info["effective_fps"]
    resume_info = detect_resume_point(output_dir, args.frame_stride,
                                      {v: k for k, v in index_mapping.items()} if index_mapping else None)

    print("\nBuilding SAM3 video predictor...")
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    overrides = apply_threshold_overrides(predictor, args)

    try:
        scan_rows, prompt_frame = None, 0
        if args.prompt_frame is not None:
            prompt_frame = args.prompt_frame
            print(f"\nUsing the given prompt frame {prompt_frame} (scan skipped)")
        elif args.scan_stride > 0:
            scan_rows = scan_prompt_frames(predictor, video_path_to_use, args.text_prompt, args.scan_stride,
                                           args.max_scan_frames, args.offload_to_cpu, index_mapping)
            best = choose_prompt_frame(scan_rows)
            print_scan(scan_rows, best)
            prompt_frame = best["frame"]
            with open(os.path.join(output_dir, "prompt_scan.json"), "w") as f:
                json.dump({"text_prompt": args.text_prompt, "scan_stride": args.scan_stride,
                           "chosen_prompt_frame": prompt_frame, "rows": scan_rows}, f, indent=2)
        else:
            print("\nScan disabled; prompt on frame 0")

        outputs, metadata = process_video_in_chunks(
            predictor=predictor, video_path=video_path_to_use, output_dir=output_dir,
            text_prompt=args.text_prompt, chunk_size=None, prompt_frame=prompt_frame,
            save_masks=True, save_visualizations=True, offload_to_cpu=args.offload_to_cpu,
            index_mapping=index_mapping, frame_stride=args.frame_stride, resume_info=resume_info)

        metadata.update({
            "prompt_frame": prompt_frame,
            "prompt_frame_original": int(index_mapping.get(prompt_frame, prompt_frame)) if index_mapping else prompt_frame,
            "prompt_scan": "prompt_scan.json" if scan_rows else None,
            "threshold_overrides": overrides,
            "frame_stride": args.frame_stride, "max_frames": args.max_frames,
            "effective_fps": effective_fps, "original_video_path": args.video_path,
        })
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        if args.save_video:
            create_output_video(output_dir, fps=effective_fps if effective_fps else metadata["fps"])
        print(f"\nDone. Prompt frame {prompt_frame}; frames with detections "
              f"{metadata['frames_with_detections']}/{metadata['frames_processed']}; "
              f"objects {metadata['object_ids']}")
    finally:
        print("\nShutting down predictor...")
        predictor.shutdown()
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
