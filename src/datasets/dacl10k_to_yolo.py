"""
Convert dacl10k polygon annotations → YOLO bbox format.
Filters to heritage-relevant classes only.

Output structure:
  data/dacl10k_yolo/
    images/{train,val}/
    labels/{train,val}/
    data.yaml
"""

import json
import shutil
from pathlib import Path

DACL_ROOT = Path("data/dacl10k/dacl10k_v2_devphase")
OUT_ROOT  = Path("data/dacl10k_yolo")

# Heritage-relevant classes only; maps dacl10k label → YOLO class id
KEEP_CLASSES = {
    "Crack":         0,
    "ACrack":        0,  # merge alligator crack → crack
    "Efflorescence": 1,
    "Spalling":      2,
    "Weathering":    3,
    "Wetspot":       4,
}

CLASS_NAMES = ["crack", "efflorescence", "spalling", "weathering", "wetspot"]

# dacl10k split → output split
SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
}


def polygon_to_bbox(points, img_w, img_h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(0, min(xs)), min(img_w, max(xs))
    y1, y2 = max(0, min(ys)), min(img_h, max(ys))
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def convert_split(dacl_split, out_split):
    ann_dir = DACL_ROOT / "annotations" / dacl_split
    img_dir = DACL_ROOT / "images" / dacl_split

    out_img = OUT_ROOT / "images" / out_split
    out_lbl = OUT_ROOT / "labels" / out_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = n_skipped = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        img_w = ann["imageWidth"]
        img_h = ann["imageHeight"]
        img_name = ann["imageName"]

        lines = []
        for shape in ann["shapes"]:
            label = shape["label"]
            if label not in KEEP_CLASSES:
                continue
            cls_id = KEEP_CLASSES[label]
            cx, cy, w, h = polygon_to_bbox(shape["points"], img_w, img_h)
            if w < 0.001 or h < 0.001:
                continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        src_img = img_dir / img_name
        if not src_img.exists():
            n_skipped += 1
            continue

        # write label (empty file = background image, still useful)
        (out_lbl / ann_path.with_suffix(".txt").name).write_text("\n".join(lines))
        shutil.copy2(src_img, out_img / img_name)

        n_images += 1
        n_boxes  += len(lines)

    return n_images, n_boxes, n_skipped


def write_yaml():
    yaml = f"""path: {OUT_ROOT.resolve()}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (OUT_ROOT / "data.yaml").write_text(yaml)


if __name__ == "__main__":
    stats = {}
    for dacl_split, out_split in SPLIT_MAP.items():
        imgs, boxes, skipped = convert_split(dacl_split, out_split)
        stats[out_split] = (imgs, boxes, skipped)
        print(f"{out_split:5s}: {imgs} images, {boxes} boxes, {skipped} skipped")

    write_yaml()
    print(f"\ndata.yaml written to {OUT_ROOT / 'data.yaml'}")
    print("Classes:", CLASS_NAMES)
