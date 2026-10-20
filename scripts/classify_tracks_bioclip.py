#!/usr/bin/env python3
"""
Species classification of SAM3 tracks with BioCLIP 2.5 (or BioCLIP 2), closed label list.

Runs AFTER process_video.py: for every tracked object (masks/obj_<k>/frame_XXXXXX.png) it cuts box
crops with context from the original video, classifies up to N crops per track with BioCLIP against the
given species names, averages the probabilities over the track and writes the result next to the masks:

    <sam3_output_dir>/species_per_track.json   per track: species, mean probability per label, votes
    <sam3_output_dir>/species_per_track.csv    same as a table
    <sam3_output_dir>/species_obj_<k>.jpg      contact sheet of the crops used, with the decision
    <sam3_output_dir>/metadata.json            gets a "species" block added

Usage:
    python scripts/classify_tracks_bioclip.py \
        --sam3_output_dir adam_brent_out/10080023/animal \
        --video_path adam_brent/LOD_7_9ott_ABa/09ott26/10080023.AVI \
        --species "roe deer,honey badger,red fox"

Notes:
    - Box crops keep the background (BioCLIP was trained on ordinary photos, not masked cut-outs).
    - A track is marked "uncertain" when its best mean probability is below --min_prob. With a closed
      list every crop gets a label even when none fits, so a low best score is the only "none of these"
      signal; add more candidate species to the list to make that signal sharper.
    - Model weights download from Hugging Face on first use (BioCLIP 2.5: ViT-H/14, ~3.9 GB).
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

DEFAULT_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"
BIOCLIP2_MODEL = "hf-hub:imageomics/bioclip-2"


def parse_args():
    p = argparse.ArgumentParser(description="Classify SAM3 tracks with BioCLIP (closed species list)")
    p.add_argument("--sam3_output_dir", required=True,
                   help="prompt sub-folder written by process_video.py (contains masks/ and metadata.json)")
    p.add_argument("--video_path", default=None,
                   help="original video or frame folder; default: taken from metadata.json")
    p.add_argument("--species", required=True, help='comma-separated label list, e.g. "roe deer,honey badger,red fox"')
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"open_clip model string (default {DEFAULT_MODEL}; "
                                                          f"use {BIOCLIP2_MODEL} for the smaller BioCLIP 2)")
    p.add_argument("--max_crops_per_track", type=int, default=24)
    p.add_argument("--margin", type=float, default=0.15, help="box margin as a fraction of the box size")
    p.add_argument("--min_box", type=int, default=32, help="skip masks whose box is smaller than this (px)")
    p.add_argument("--min_prob", type=float, default=0.5, help="below this best mean probability a track is 'uncertain'")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_crops", action="store_true", help="also save every crop used as a JPEG")
    return p.parse_args()


# ---------------------------------------------------------------------------
# reading SAM3 output
# ---------------------------------------------------------------------------

def list_tracks(sam3_output_dir):
    """{obj_id: {frame_number: mask_path}} from masks/obj_<k>/frame_XXXXXX.png"""
    tracks = {}
    for d in sorted(glob.glob(os.path.join(sam3_output_dir, "masks", "obj_*"))):
        m = re.match(r"obj_(\d+)$", os.path.basename(d))
        if not m:
            continue
        frames = {}
        for p in glob.glob(os.path.join(d, "frame_*.png")):
            fm = re.match(r"frame_(\d+)\.png", os.path.basename(p))
            if fm:
                frames[int(fm.group(1))] = p
        if frames:
            tracks[int(m.group(1))] = frames
    return tracks


def pick_frames(frame_numbers, n):
    frame_numbers = sorted(frame_numbers)
    if len(frame_numbers) <= n:
        return frame_numbers
    idx = np.linspace(0, len(frame_numbers) - 1, n).round().astype(int)
    return [frame_numbers[i] for i in sorted(set(idx))]


def frame_reader(video_path):
    """Yield (frame_number, frame_bgr) in order, for a video file or a folder of numbered JPEGs."""
    if os.path.isdir(video_path):
        paths = glob.glob(os.path.join(video_path, "*.jpg")) + glob.glob(os.path.join(video_path, "*.png"))
        def key(p):
            m = re.search(r"(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        for p in sorted(paths, key=key):
            yield key(p), cv2.imread(p)
        return
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def box_crop(frame, mask, margin, min_box):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if max(bw, bh) < min_box:
        return None
    side = int(max(bw, bh) * (1 + 2 * margin))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    h, w = frame.shape[:2]
    canvas = np.zeros((side, side, 3), np.uint8)
    ix0, iy0, ix1, iy1 = max(sx0, 0), max(sy0, 0), min(sx0 + side, w), min(sy0 + side, h)
    canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = frame[iy0:iy1, ix0:ix1]
    return canvas


def collect_crops(sam3_output_dir, video_path, max_per_track, margin, min_box):
    """Decode the video once; return {obj_id: [(frame_number, crop_bgr), ...]}"""
    tracks = list_tracks(sam3_output_dir)
    if not tracks:
        sys.exit(f"no masks found under {sam3_output_dir}/masks")
    wanted = defaultdict(list)  # frame_number -> [(obj_id, mask_path)]
    for obj, frames in tracks.items():
        for fn in pick_frames(frames.keys(), max_per_track):
            wanted[fn].append((obj, frames[fn]))
    last = max(wanted)
    crops = defaultdict(list)
    for fn, frame in frame_reader(video_path):
        if fn > last:
            break
        if fn not in wanted or frame is None:
            continue
        for obj, mask_path in wanted[fn]:
            mask = cv2.imread(mask_path, 0)
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            c = box_crop(frame, mask, margin, min_box)
            if c is not None:
                crops[obj].append((fn, c))
    return tracks, crops


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def load_classifier(species, model, device):
    import torch
    from bioclip import CustomLabelsClassifier
    print(f"Loading BioCLIP model {model} on {device} ...")
    # pybioclip wraps the model in torch.compile unconditionally. Compiling a ViT-H for a few hundred
    # crops costs more than it saves, and it fails outright on torch < 2.4 with Python 3.12, so skip it.
    real_compile = torch.compile
    torch.compile = lambda m, *a, **k: m
    try:
        return CustomLabelsClassifier(cls_ary=species, model_str=model, device=device)
    finally:
        torch.compile = real_compile


def classify_crops(classifier, species, crops_bgr, batch_size):
    """Return (n_crops, n_species) probability matrix in the order of `species`."""
    from PIL import Image
    images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    preds = classifier.predict(images, batch_size=batch_size)  # flat list: n_crops * n_species dicts
    probs = np.zeros((len(images), len(species)), np.float32)
    col = {s: i for i, s in enumerate(species)}
    per_image = defaultdict(dict)
    for p in preds:
        per_image[p["file_name"]][p["classification"]] = p["score"]
    for i, key in enumerate(per_image.keys()):
        for s, v in per_image[key].items():
            probs[i, col[s.strip()]] = v
    return probs


def summarise(species, probs, min_prob):
    mean = probs.mean(0)
    votes = np.bincount(probs.argmax(1), minlength=len(species))
    best = int(mean.argmax())
    return dict(
        species=species[best] if mean[best] >= min_prob else "uncertain",
        best_label=species[best],
        confidence=float(mean[best]),
        mean_prob={s: float(v) for s, v in zip(species, mean)},
        votes={s: int(v) for s, v in zip(species, votes)},
        n_crops=int(len(probs)),
    )


def contact_sheet(crops, labels_per_crop, title, path, tile=160, cols=8):
    sel = crops[:cols * 2]
    rows = int(np.ceil(len(sel) / cols))
    canvas = np.full((rows * tile + 28, cols * tile, 3), 30, np.uint8)
    for i, ((fn, c), lab) in enumerate(zip(sel, labels_per_crop)):
        r, k = divmod(i, cols)
        t = cv2.resize(c, (tile, tile))
        cv2.putText(t, f"f{fn} {lab}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        canvas[28 + r * tile:28 + (r + 1) * tile, k * tile:(k + 1) * tile] = t
    cv2.putText(canvas, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def run(sam3_output_dir, video_path, species, model=DEFAULT_MODEL, max_crops_per_track=24, margin=0.15,
        min_box=32, min_prob=0.5, batch_size=8, device="cuda", save_crops=False, classifier=None):
    species = [s.strip() for s in species if s.strip()]
    meta_path = os.path.join(sam3_output_dir, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    if video_path is None:
        video_path = meta.get("original_video_path") or meta.get("video_path")
        if not video_path or not os.path.exists(video_path):
            sys.exit("pass --video_path: metadata has no usable video path")

    print(f"Collecting crops from {video_path} ...")
    tracks, crops = collect_crops(sam3_output_dir, video_path, max_crops_per_track, margin, min_box)
    print(f"  {len(tracks)} track(s), crops per track: " + ", ".join(f"obj_{o}={len(c)}" for o, c in sorted(crops.items())))

    if classifier is None:
        classifier = load_classifier(species, model, device)

    results = {}
    for obj in sorted(tracks):
        if not crops.get(obj):
            results[obj] = dict(species="no usable crops", best_label=None, confidence=0.0, mean_prob={},
                                votes={}, n_crops=0, frames=sorted(tracks[obj]))
            print(f"  obj_{obj}: no usable crops (boxes smaller than --min_box {min_box} px or frames not found)")
            continue
        probs = classify_crops(classifier, species, [c for _, c in crops[obj]], batch_size)
        r = summarise(species, probs, min_prob)
        r["frames"] = [fn for fn, _ in crops[obj]]
        r["per_crop_best"] = [species[i] for i in probs.argmax(1)]
        results[obj] = r
        title = f"obj_{obj}: {r['species']}  (mean p={r['confidence']:.2f}, votes {r['votes']})"
        contact_sheet(crops[obj], [f"{species[i][:10]} {p[i]:.2f}" for i, p in zip(probs.argmax(1), probs)],
                      title, os.path.join(sam3_output_dir, f"species_obj_{obj}.jpg"))
        if save_crops:
            d = os.path.join(sam3_output_dir, "species_crops", f"obj_{obj}")
            os.makedirs(d, exist_ok=True)
            for fn, c in crops[obj]:
                cv2.imwrite(os.path.join(d, f"frame_{fn:06d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  obj_{obj}: {r['species']:<14} mean p {r['confidence']:.2f}  "
              + "  ".join(f"{s}={r['mean_prob'][s]:.2f}" for s in species) + f"  ({r['n_crops']} crops)")

    out = dict(model=model, species_list=species, min_prob=min_prob, max_crops_per_track=max_crops_per_track,
               video_path=video_path, tracks={f"obj_{o}": r for o, r in results.items()})
    with open(os.path.join(sam3_output_dir, "species_per_track.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(sam3_output_dir, "species_per_track.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "species", "confidence", "n_crops"] + [f"p({s})" for s in species])
        for o, r in results.items():
            w.writerow([f"obj_{o}", r["species"], f"{r['confidence']:.3f}", r["n_crops"]]
                       + [f"{r['mean_prob'].get(s, 0):.3f}" for s in species])
    meta["species"] = {f"obj_{o}": dict(species=r["species"], confidence=r["confidence"], n_crops=r["n_crops"])
                       for o, r in results.items()}
    meta["species_model"] = model
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwritten: species_per_track.json / .csv and species_obj_<k>.jpg in {sam3_output_dir}")
    return out


def main():
    args = parse_args()
    run(args.sam3_output_dir, args.video_path, args.species.split(","), model=args.model,
        max_crops_per_track=args.max_crops_per_track, margin=args.margin, min_box=args.min_box,
        min_prob=args.min_prob, batch_size=args.batch_size, device=args.device, save_crops=args.save_crops)


if __name__ == "__main__":
    main()
