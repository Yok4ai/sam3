# Share-of-Shelf Auto-Annotation Pipeline

## Goal

Auto-annotate a flat folder of retail shelf images using SAM3 image inference. Output is YOLO detect format (bounding boxes) ready for Ultralytics training.

## Dataset

- **Input:** `/home/mkultra/Documents/sam3/dataset/SOS/` — 5000 flat JPEGs
- **Output:** `/home/mkultra/Documents/sam3/dataset/SOS_labels/` — one `.txt` per image
- **Class:** single class `0` = `product`

## Script

`scripts/annotate_shelf.py`

```
python scripts/annotate_shelf.py \
  --images dataset/SOS \
  --output dataset/SOS_labels \
  --batch-size 8 \
  --prompt "product"
```

## Pipeline

1. Glob `*.jpg` / `*.png` from `--images`
2. Skip images where corresponding `.txt` already exists in `--output` (resume-safe)
3. Load batch of N images as PIL → `set_image_batch` → shared backbone forward pass
4. For each image in batch: `set_text_prompt(state, prompt)` → get `boxes`
5. Convert boxes from absolute `[x1 y1 x2 y2]` to normalized `cx cy w h`
6. Write `0 cx cy w h` lines to `.txt`; write empty `.txt` if no detections
7. tqdm progress bar advances by batch size

## Output Format

YOLO detect format — one line per detection:
```
0 cx cy w h
```
All values normalized `[0, 1]`. Empty file = no detections (YOLO expects the file to exist).

## Key Decisions

- No confidence filtering at annotation time — tune threshold during YOLO training
- Batch size 8 default — safe for most GPUs at 1008px resolution, override via CLI
- Labels dir is sibling to images dir to keep dataset portable
- No train/val split at this stage — inspect annotations first, split manually

## Post-annotation

After running the script:
1. Spot-check a sample with any labeling tool (LabelImg, Roboflow, etc.)
2. Create `data.yaml` pointing to split folders
3. Run `yolo train`
