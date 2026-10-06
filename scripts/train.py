"""Train YOLO11n, choose weights on validation only, and export the app model."""
from pathlib import Path
import argparse
import json
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", type=Path, default=ROOT / "dataset/data.yaml")
    args = p.parse_args()
    if args.data.resolve() == (ROOT / "dataset/data.yaml").resolve():
        from unpack_dataset import ensure_dataset
        ensure_dataset()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.data.read_text())
    config["path"] = str(args.data.resolve().parent)
    generated = ROOT / "reports/runtime_data.yaml"
    generated.write_text(yaml.safe_dump(config))
    model = YOLO("yolo11n.pt")
    model.train(data=str(generated), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, seed=42,
                deterministic=True, optimizer="AdamW", lr0=0.001,
                freeze=10, patience=args.epochs, cache=False,
                degrees=10, fliplr=.5, flipud=0, hsv_v=.3, hsv_s=.4,
                mosaic=1.0, close_mosaic=5, scale=.3, translate=.1,
                project=str(ROOT / "runs"), name="stocksense", exist_ok=False,
                plots=True)
    best = Path(model.trainer.best)
    shutil.copy2(best, ROOT / "models/best.pt")
    trained = YOLO(str(ROOT / "models/best.pt"))
    exported = trained.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=False, dynamic=False, nms=False)
    if Path(exported).resolve() != (ROOT / "models/best.onnx").resolve():
        shutil.copy2(exported, ROOT / "models/best.onnx")
    names = [trained.names[i] for i in range(len(trained.names))]
    (ROOT / "models/metadata.json").write_text(json.dumps(dict(
        architecture="YOLO11n", classes=names, input_size=args.imgsz,
        training_epochs=args.epochs, batch_size=args.batch, seed=42,
        initialization="Ultralytics YOLO11n COCO pretrained weights",
        dataset="500-image five-category Freiburg grocery detection subset",
        confidence_default=.25, iou_default=.45,
        selection="Best validation mAP; test set not used for model selection",
    ), indent=2))
    for filename in ("results.csv", "results.png", "args.yaml"):
        src = best.parents[1] / filename
        if src.exists():
            shutil.copy2(src, ROOT / "reports" / filename)
    print("Saved trained weights and ONNX model to models/")


if __name__ == "__main__":
    main()
