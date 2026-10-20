#!/usr/bin/env python3
"""
SAM3 + BioCLIP in one command: segment and track every animal with a generic prompt, then name the
species of each track from a closed list.

Step 1  process_video.py with --text_prompt (default "animal"): masks per tracked object.
Step 2  classify_tracks_bioclip.py: box crops of each track from the original video, BioCLIP 2.5
        probabilities for the given species, averaged per track; results in species_per_track.json/.csv,
        one contact sheet per track, and a "species" block in metadata.json.

Usage:
    python scripts/process_video_bioclip.py \
        --video_path adam_brent/LOD_7_9ott_ABa/09ott26/10080023.AVI \
        --output_dir adam_brent_out/10080023 \
        --text_prompt animal --species "roe deer,honey badger,red fox" \
        --max_frames 300 --save_video

All process_video.py options (--max_frames, --frame_stride, --prompt_frame, --gpus, --offload_to_cpu,
--save_video) are accepted and passed through. Add --classify_only to skip step 1 when the SAM3 output
already exists.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_tracks_bioclip as ctb  # noqa: E402
import process_video  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 tracking + BioCLIP species per track")
    p.add_argument("--video_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--text_prompt", default="animal", help="SAM3 prompt for step 1 (default: animal)")
    p.add_argument("--species", required=True, help='closed label list, e.g. "roe deer,honey badger,red fox"')
    p.add_argument("--model", default=ctb.DEFAULT_MODEL, help="BioCLIP model string")
    p.add_argument("--max_crops_per_track", type=int, default=24)
    p.add_argument("--min_prob", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--classify_only", action="store_true", help="skip SAM3, classify existing output")
    # pass-through to process_video.py
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--prompt_frame", type=int, default=0)
    p.add_argument("--chunk_size", type=int, default=None, help="ignored (kept for old commands)")
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--offload_to_cpu", action="store_true", default=False)
    p.add_argument("--save_video", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    sam3_dir = process_video.create_prompt_subdir(args.output_dir, args.text_prompt)

    if not args.classify_only:
        argv = ["process_video.py", "--video_path", args.video_path, "--output_dir", args.output_dir,
                "--text_prompt", args.text_prompt, "--prompt_frame", str(args.prompt_frame),
                "--frame_stride", str(args.frame_stride)]
        if args.max_frames is not None:
            argv += ["--max_frames", str(args.max_frames)]
        if args.gpus:
            argv += ["--gpus", args.gpus]
        if args.offload_to_cpu:
            argv += ["--offload_to_cpu"]
        if args.save_video:
            argv += ["--save_video"]
        print("=" * 60 + "\nSTEP 1: SAM3 tracking\n" + "=" * 60)
        saved_argv = sys.argv
        sys.argv = argv
        try:
            process_video.main()  # shuts the SAM3 predictor down before returning, freeing the GPU
        finally:
            sys.argv = saved_argv

    if not os.path.isdir(os.path.join(sam3_dir, "masks")):
        sys.exit(f"no masks in {sam3_dir}; SAM3 found nothing to track (try another --text_prompt or --prompt_frame)")

    print("\n" + "=" * 60 + "\nSTEP 2: BioCLIP species per track\n" + "=" * 60)
    ctb.run(sam3_dir, args.video_path, args.species.split(","), model=args.model,
            max_crops_per_track=args.max_crops_per_track, min_prob=args.min_prob, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
