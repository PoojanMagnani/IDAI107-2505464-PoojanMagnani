"""Measure the exported app model on held-out images, including count accuracy."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from stocksense.detector import Detector, iou_one, annotate
from stocksense.logic import summarize

ROOT = Path(__file__).resolve().parents[1]


def ground_truth(path, size, names):
    width, height = size
    result = []
    for row in path.read_text().splitlines():
        cls, x, y, w, h = map(float, row.split())
        result.append(dict(category=names[int(cls)], box=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]))
    return result


def score(predictions, truth, names):
    matches = set()
    tp, fp, fn = Counter(), Counter(), Counter()
    for d in sorted(predictions,key=lambda d:-d["confidence"]):
        candidates = [i for i,g in enumerate(truth) if g["category"] == d["category"] and i not in matches]
        overlaps = iou_one(np.array(d["box"]), np.array([truth[i]["box"] for i in candidates]))
        if len(overlaps) and overlaps.max() >= .5:
            matches.add(candidates[int(overlaps.argmax())]); tp[d["category"]] += 1
        else:
            fp[d["category"]] += 1
    for i,g in enumerate(truth):
        if i not in matches:
            fn[g["category"]] += 1
    p_count, g_count = Counter(d["category"] for d in predictions), Counter(g["category"] for g in truth)
    errors = {n: abs(p_count[n]-g_count[n]) for n in names}
    return tp, fp, fn, errors, p_count, g_count


def run_variant(detector, paths, variant="original", save=False):
    names = detector.names
    total_tp, total_fp, total_fn, error_sums = Counter(), Counter(), Counter(), Counter()
    records = []
    exact = 0
    matrix = np.zeros((len(names)+1,len(names)+1), dtype=int)
    for path in paths:
        image = Image.open(path).convert("RGB")
        truth = ground_truth(ROOT / "dataset/labels/test" / (path.stem+".txt"), image.size,names)
        if variant == "low_light":
            image = ImageEnhance.Brightness(image).enhance(.45)
        if variant == "horizontal_flip":
            image = ImageOps.mirror(image)
            for g in truth:
                x1,y1,x2,y2 = g["box"]
                g["box"] = [image.width-x2,y1,image.width-x1,y2]
        if variant == "cluttered_subset" and len(truth) < 3:
            continue
        detections = detector.predict(image,.25,.45,False)
        tp, fp, fn, errors, predicted, actual = score(detections, truth, names)
        total_tp.update(tp); total_fp.update(fp); total_fn.update(fn); error_sums.update(errors)
        exact += int(sum(errors.values()) == 0)
        row = dict(image=path.name, ground_truth_items=sum(actual.values()), detected_items=sum(predicted.values()),
                   absolute_category_error=sum(errors.values()), exact_counts=sum(errors.values())==0)
        for name in names:
            row[f"{name}_true"] = actual[name]; row[f"{name}_predicted"] = predicted[name]
        records.append(row)
        # Class-agnostic one-to-one IoU matching for a confusion matrix including background.
        used = set()
        for detection in sorted(detections,key=lambda d:-d["confidence"]):
            candidates = [i for i in range(len(truth)) if i not in used]
            overlaps = iou_one(np.array(detection["box"]),np.array([truth[i]["box"] for i in candidates]))
            if len(overlaps) and overlaps.max() >= .5:
                j=candidates[int(overlaps.argmax())]; used.add(j)
                matrix[names.index(truth[j]["category"]),names.index(detection["category"])] += 1
            else:
                matrix[-1,names.index(detection["category"])] += 1
        for j,g in enumerate(truth):
            if j not in used:
                matrix[names.index(g["category"]),-1] += 1
        if save:
            out = ROOT / "reports/test_predictions"
            out.mkdir(exist_ok=True)
            (out / (path.stem+".json")).write_text(json.dumps(detections,indent=2))
    n = len(records)
    per_class = []
    for name in names:
        tp,fp,fn = total_tp[name],total_fp[name],total_fn[name]
        per_class.append(dict(category=name,precision=tp/(tp+fp) if tp+fp else 0,
                              recall=tp/(tp+fn) if tp+fn else 0,tp=tp,fp=fp,fn=fn,
                              count_mae=error_sums[name]/n if n else 0))
    tp,fp,fn = sum(total_tp.values()),sum(total_fp.values()),sum(total_fn.values())
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    return dict(variant=variant,images=n,precision=precision,recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                count_mae=sum(error_sums.values())/(n*len(names)) if n else 0,
                exact_count_rate=exact/n if n else 0,per_class=per_class), records, matrix


def main():
    from unpack_dataset import ensure_dataset
    ensure_dataset()
    detector = Detector(ROOT / "models/best.onnx", ROOT / "models/metadata.json")
    paths = sorted((ROOT / "dataset/images/test").glob("*.png"))
    if not paths:
        raise ValueError("No test images found")
    result, records, matrix = run_variant(detector,paths,save=True)
    result["stress_tests"] = []
    for variant in ["low_light","horizontal_flip","cluttered_subset"]:
        stress,_,_ = run_variant(detector,paths,variant)
        result["stress_tests"].append({k:v for k,v in stress.items() if k != "per_class"})
    result["protocol"] = dict(split="test",confidence=.25,nms_iou=.45,match_iou=.5,tiled=False,
                               count_mae="Mean absolute count error across all five categories and test images",
                               selection="Test results measured after selecting the best model on validation.")
    (ROOT / "reports/evaluation.json").write_text(json.dumps(result,indent=2))
    with (ROOT / "reports/counting_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6))
    ax.imshow(matrix,cmap="Blues")
    names=detector.names+["background"]
    ax.set(xticks=range(6),xticklabels=names,yticks=range(6),yticklabels=names,xlabel="Predicted category",ylabel="True category",title="Held-out test confusion matrix · IoU ≥ 0.5")
    for i in range(6):
        for j in range(6):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>matrix.max()/2 else "#17243B")
    fig.tight_layout();fig.savefig(ROOT / "reports/confusion_matrix.png",dpi=160);plt.close(fig)
    (ROOT / "reports/confusion_matrix.json").write_text(json.dumps(dict(labels=names,rows="true",columns="predicted",values=matrix.tolist()),indent=2))
    # Also retain standard detector mAP; this is a separate evaluation protocol.
    import yaml
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    config=yaml.safe_load((ROOT/"dataset/data.yaml").read_text())
    config["path"]=str(ROOT/"dataset")
    runtime=ROOT/"reports/runtime_data.yaml"
    runtime.write_text(yaml.safe_dump(config))
    standard=YOLO(str(ROOT/"models/best.pt")).val(data=str(runtime),split="test",
        imgsz=detector.size,batch=16,device="cpu",workers=0,plots=True,
        project=str(ROOT/"runs"),name="test_evaluation")
    (ROOT/"reports/ultralytics_test_metrics.json").write_text(json.dumps(
        {k:float(v) for k,v in standard.results_dict.items()},indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"stress_tests","per_class"}},indent=2))


if __name__ == "__main__":
    main()
