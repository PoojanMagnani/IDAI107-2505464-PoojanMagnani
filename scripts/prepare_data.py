"""Convert a balanced Freiburg grocery subset from Pascal VOC to YOLO format.

Run from the repository root. Source annotations are retained for audit.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
import yaml

CLASSES = ["cereal", "coffee", "juice", "milk", "water"]


def prepare(source: Path, output: Path, per_class: int = 100, seed: int = 42):
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new directory.")
    rng = random.Random(seed)
    candidates = {}
    seen = set()
    for name in CLASSES:
        items = []
        for image in sorted((source / "images" / name).glob("*.png")):
            annotation = source / "annotations" / name / (image.stem + ".xml")
            if not annotation.exists():
                continue
            with Image.open(image) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in seen:
                continue
            root = ET.parse(annotation).getroot()
            boxes = []
            for obj in root.findall("object"):
                label = obj.findtext("name", "").strip().lower()
                if label not in CLASSES:
                    continue
                box = obj.find("bndbox")
                x1, y1, x2, y2 = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Invalid box in {annotation}")
                boxes.append((CLASSES.index(label), (x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h))
            if not boxes or not any(b[0] == CLASSES.index(name) for b in boxes):
                continue
            seen.add(digest)
            items.append((image, annotation, digest, boxes))
        rng.shuffle(items)
        if len(items) < per_class:
            raise ValueError(f"{name}: only {len(items)} unique labeled images, need {per_class}")
        candidates[name] = items[:per_class]
    rows = []
    for name, items in candidates.items():
        for i, (image, annotation, digest, boxes) in enumerate(items):
            split = "train" if i < int(per_class*.7) else "val" if i < int(per_class*.85) else "test"
            for folder in ("images", "labels", "annotations"):
                (output / folder / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, output / "images" / split / image.name)
            shutil.copy2(annotation, output / "annotations" / split / annotation.name)
            (output / "labels" / split / (image.stem + ".txt")).write_text(
                "\n".join(f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}" for c,x,y,w,h in boxes) + "\n")
            rows.append(dict(file=image.name, category=name, split=split, boxes=len(boxes), pixel_sha256=digest))
    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (output / "data.yaml").write_text(yaml.safe_dump(dict(path=".", train="images/train", val="images/val", test="images/test", names=CLASSES)))
    summary = {s: {c: sum(r["split"] == s and r["category"] == c for r in rows) for c in CLASSES} for s in ("train", "val", "test")}
    (output / "split_summary.json").write_text(json.dumps(dict(seed=seed, images=len(rows), counts=summary, exact_pixel_duplicates_removed=True), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="The downloaded source repository's dataset folder")
    p.add_argument("--output", type=Path, default=Path("dataset"))
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare(args.source, args.output, args.per_class, args.seed)
