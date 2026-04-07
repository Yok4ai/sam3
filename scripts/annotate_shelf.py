"""
Annotate a flat folder of shelf images with SAM3 via Ultralytics, writing YOLO detect labels.

Requires:
    pip install -U ultralytics
    sam3.pt in working directory (download from HuggingFace after requesting access)

Usage:
    python scripts/annotate_shelf.py \
        --images dataset/SOS \
        --output dataset/SOS_labels \
        --prompt "product on shelf" \
        [--n 100] [--visualize] [--conf 0.25]
"""

import argparse
import glob
import os

from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor


def save_viz(img, boxes_xyxy_cls, viz_path):
    """Save image with colored bounding box overlays."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes_xyxy_cls:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    img.save(viz_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",    required=True,  help="Folder of input images")
    parser.add_argument("--output",    required=True,  help="Folder for YOLO .txt labels")
    parser.add_argument("--model",     default="sam3.pt", help="Path to sam3.pt weights")
    parser.add_argument("--prompt",    default="product on shelf", help="Text prompt")
    parser.add_argument("--conf",      type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-area",  type=float, default=0.5,  help="Drop boxes covering more than this fraction of the image")
    parser.add_argument("--n",         type=int,   default=None, help="Limit to first N images")
    parser.add_argument("--visualize", action="store_true", help="Save overlay images to <output>_viz/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    viz_dir = args.output.rstrip("/") + "_viz"
    if args.visualize:
        os.makedirs(viz_dir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(args.images, ext)))
    all_images = sorted(all_images)

    todo = []
    for img_path in all_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.output, stem + ".txt")
        if not os.path.exists(label_path):
            todo.append((img_path, label_path))

    if args.n is not None:
        todo = todo[: args.n]

    print(f"{len(all_images)} images found, {len(todo)} to process")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading {args.model}...")
    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
        half=True,
        save=False,
        verbose=False,
    ))

    with tqdm(total=len(todo), unit="img") as bar:
        for img_path, label_path in todo:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((896, 896), Image.LANCZOS)
            predictor.set_image(img)
            results = predictor(text=[args.prompt])

            r = results[0]
            lines = []
            kept_xyxy = []
            if r.boxes is not None and len(r.boxes):
                for xywhn_row, xyxy_row in zip(r.boxes.xywhn.tolist(), r.boxes.xyxy.tolist()):
                    cx, cy, w, h = xywhn_row[:4]
                    if w * h > args.max_area:
                        continue
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    kept_xyxy.append(xyxy_row)

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            if args.visualize and kept_xyxy:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                save_viz(img, kept_xyxy, os.path.join(viz_dir, stem + ".jpg"))

            bar.update(1)

    print(f"Done. Labels written to {args.output}/")
    if args.visualize:
        print(f"Overlays written to {viz_dir}/")


if __name__ == "__main__":
    main()
