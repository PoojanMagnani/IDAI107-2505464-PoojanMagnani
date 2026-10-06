"""YOLO11 ONNX inference. No PyTorch or network download is needed in the app.

Model output: [batch, 4 + number_of_classes, anchors]. The first four values
are center-x, center-y, width and height in letterboxed input pixels.
"""
from pathlib import Path
from io import BytesIO
import json
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 24_000_000


def read_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose an image smaller than 12 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Please choose a JPEG, PNG, or WebP image.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Please resize the image to fewer than 24 million pixels.")
                result = ImageOps.exif_transpose(image).convert("RGB")
                result.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                return result.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("This image could not be opened. Try a different JPEG, PNG, or WebP.") from exc


def letterbox(image: Image.Image, size: int):
    scale = min(size / image.width, size / image.height)
    width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
    left, top = (size-width)//2, (size-height)//2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, scale, left, top


def iou_one(box, boxes):
    if len(boxes) == 0:
        return np.array([], dtype=float)
    tl = np.maximum(box[:2], boxes[:, :2])
    br = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0, br-tl)
    intersection = wh[:, 0]*wh[:, 1]
    area1 = max(0, box[2]-box[0])*max(0, box[3]-box[1])
    area2 = np.maximum(0, boxes[:, 2]-boxes[:, 0])*np.maximum(0, boxes[:, 3]-boxes[:, 1])
    return intersection / np.maximum(area1+area2-intersection, 1e-9)


def nms(detections: list[dict], threshold: float = .45, max_items: int = 300):
    """Class-aware non-maximum suppression removes overlapping duplicate boxes."""
    kept = []
    for category in sorted({d["category"] for d in detections}):
        group = sorted((d for d in detections if d["category"] == category), key=lambda d: -d["confidence"])
        while group and len(kept) < max_items:
            winner = group.pop(0)
            kept.append(winner)
            if group:
                overlap = iou_one(np.array(winner["box"]), np.array([d["box"] for d in group]))
                group = [d for d, overlap_value in zip(group, overlap) if overlap_value <= threshold]
    return sorted(kept, key=lambda d: -d["confidence"])[:max_items]


class Detector:
    def __init__(self, model_path: Path, metadata_path: Path):
        import onnxruntime as ort
        self.metadata = json.loads(metadata_path.read_text())
        self.names = self.metadata["classes"]
        self.size = self.metadata["input_size"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _predict(self, image, confidence):
        tensor, scale, left, top = letterbox(image, self.size)
        pred = self.session.run(None, {self.input_name: tensor})[0][0].T
        if pred.shape[1] != 4 + len(self.names):
            raise ValueError("Model output and category metadata do not match.")
        class_ids = pred[:, 4:].argmax(axis=1)
        scores = pred[np.arange(len(pred)), class_ids+4]
        detections = []
        for row, class_id, score in zip(pred[scores >= confidence], class_ids[scores >= confidence], scores[scores >= confidence]):
            cx, cy, w, h = row[:4]
            box = [(cx-w/2-left)/scale, (cy-h/2-top)/scale, (cx+w/2-left)/scale, (cy+h/2-top)/scale]
            box = [float(np.clip(box[0], 0, image.width)), float(np.clip(box[1], 0, image.height)),
                   float(np.clip(box[2], 0, image.width)), float(np.clip(box[3], 0, image.height))]
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(dict(category=self.names[int(class_id)], confidence=float(score), box=box))
        return detections

    def predict(self, image: Image.Image, confidence=.25, iou=.45, tiled=False):
        if not 0 < confidence <= 1 or not 0 < iou <= 1:
            raise ValueError("Confidence and IoU must be between zero and one.")
        if not tiled or max(image.size) <= 448:
            return nms(self._predict(image, confidence), iou)
        # Overlapping crops retain more small-product detail than resizing a wide shelf.
        # Tile results are merged once in original-image coordinates.
        tile = 448
        stride = 336
        def starts(length):
            if length <= tile:
                return [0]
            return sorted(set(list(range(0, length-tile+1, stride)) + [length-tile]))
        detections = []
        for y in starts(image.height):
            for x in starts(image.width):
                crop = image.crop((x, y, min(x+tile, image.width), min(y+tile, image.height)))
                for d in self._predict(crop, confidence):
                    d["box"] = [d["box"][0]+x, d["box"][1]+y, d["box"][2]+x, d["box"][3]+y]
                    detections.append(d)
        return nms(detections, iou)


def annotate(image, detections, rows):
    from stocksense.logic import STATUS_COLORS
    # Draw legible labels on a larger display canvas for small source images.
    # This changes only presentation, never inference or count coordinates.
    render_scale = max(1.0, min(4.0, 800 / max(image.size)))
    result = image.resize((round(image.width*render_scale),round(image.height*render_scale)),Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(result)
    font_size = max(13, round(max(result.size) / 55))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    status = {r["category"]: r["status"] for r in rows}
    for index, d in enumerate(detections, 1):
        color = STATUS_COLORS[status[d["category"]]]
        x1, y1, x2, y2 = [v*render_scale for v in d["box"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, font_size//7))
        label = f"{index} {d['category'].title()} {d['confidence']:.0%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = max(0, min(x1, result.width-text_width-8))
        y = max(0, y1-text_height-10)
        draw.rectangle((x,y,x+text_width+8,y+text_height+8), fill=color)
        draw.text((x+4,y+3-bbox[1]),label,fill="#101827",font=font)
    return result


def png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
