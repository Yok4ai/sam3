"""
Annotate a flat folder of shelf images with SAM3 via Ultralytics, writing YOLO detect labels.

Requires:
    pip install -U ultralytics
    sam3.pt in working directory (download from HuggingFace after requesting access)

Usage:
    python scripts/annotate_shelf.py \
        --images dataset/SOS \
        --output dataset/SOS_labels \
        --prompts "bottle" "box" "tube" "can" \
        --garbage "shelf" "wall" "floor" \
        [--n 100] [--visualize] [--conf 0.25]
"""

import argparse
import glob
import os

from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor

# viz colors: green for class 0 (product), red for class 1 (garbage)
VIZ_COLORS = {0: (0, 200, 0), 1: (255, 0, 0)}


def save_viz(img, boxes_xyxy_cls, viz_path):
    """Save image with colored bounding box overlays. boxes_xyxy_cls = [(x1,y1,x2,y2,cls), ...]"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, cls in boxes_xyxy_cls:
        color = VIZ_COLORS.get(cls, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    img.save(viz_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",   required=True,  help="Folder of input images")
    parser.add_argument("--output",   required=True,  help="Folder for YOLO .txt labels")
    parser.add_argument("--model",    default="sam3.pt", help="Path to sam3.pt weights")
    parser.add_argument("--prompts",  nargs="+", default=["product on shelf"], help="Class 0 prompts")
    parser.add_argument("--garbage",  nargs="*", default=[], help="Class 1 prompts (garbage/background)")
    parser.add_argument("--conf",     type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-area", type=float, default=0.5,  help="Drop boxes covering more than this fraction of the image")
    parser.add_argument("--n",        type=int,   default=None, help="Limit to first N images")
    parser.add_argument("--visualize", action="store_true", help="Save overlay images to <output>_viz/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    viz_dir = args.output.rstrip("/") + "_viz"
    if args.visualize:
        os.makedirs(viz_dir, exist_ok=True)

    # build prompt list + class index mapping
    all_prompts = args.prompts + args.garbage
    prompt_class = {p: 0 for p in args.prompts}
    prompt_class.update({p: 1 for p in args.garbage})

    print(f"Class 0 prompts: {args.prompts}")
    if args.garbage:
        print(f"Class 1 (garbage) prompts: {args.garbage}")

    # collect images, skip already annotated
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
            results = predictor(text=all_prompts)

            lines = []
            viz_boxes = []

            if results and results[0].boxes is not None and len(results[0].boxes):
                r = results[0]
                # results[0] contains all boxes from all prompts
                # r.boxes.cls tells us which prompt index each box came from
                for i, (xywhn_row, xyxy_row) in enumerate(
                    zip(r.boxes.xywhn.tolist(), r.boxes.xyxy.tolist())
                ):
                    cx, cy, w, h = xywhn_row[:4]
                    if w * h > args.max_area:
                        continue

                    # cls index maps back to our prompt order
                    prompt_idx = int(r.boxes.cls[i].item())
                    prompt_text = all_prompts[prompt_idx] if prompt_idx < len(all_prompts) else all_prompts[0]
                    cls = prompt_class.get(prompt_text, 0)

                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    viz_boxes.append((*xyxy_row[:4], cls))

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            if args.visualize and viz_boxes:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                save_viz(img, viz_boxes, os.path.join(viz_dir, stem + ".jpg"))

            bar.update(1)

    print(f"Done. Labels written to {args.output}/")
    if args.visualize:
        print(f"Overlays written to {viz_dir}/")


if __name__ == "__main__":
    main()
