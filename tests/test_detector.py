from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from stocksense.detector import read_image, letterbox, nms, Detector, iou_one

ROOT = Path(__file__).resolve().parents[1]


def test_bad_image_is_rejected():
    with pytest.raises(ValueError):
        read_image(b"this is not an image")


def test_grayscale_and_alpha_are_converted_to_rgb():
    for mode in ["L","RGBA"]:
        buffer=BytesIO();Image.new(mode,(20,10)).save(buffer,format="PNG")
        assert read_image(buffer.getvalue()).mode == "RGB"


def test_letterbox_preserves_aspect_and_normalizes():
    tensor,scale,left,top=letterbox(Image.new("RGB",(200,100),"white"),224)
    assert tensor.shape == (1,3,224,224)
    assert tensor.dtype == np.float32 and tensor.max() <= 1
    assert scale == 1.12 and left == 0 and top == 56


def test_nms_suppresses_same_class_only():
    detections=[dict(category="milk",confidence=.9,box=[0,0,10,10]),
                dict(category="milk",confidence=.8,box=[1,1,10,10]),
                dict(category="water",confidence=.7,box=[0,0,10,10])]
    assert len(nms(detections,.45)) == 2


def test_real_model_runs_with_valid_boxes():
    detector=Detector(ROOT/"models/best.onnx",ROOT/"models/metadata.json")
    path=next((ROOT/"dataset/images/test").glob("*.png"))
    im=Image.open(path).convert("RGB")
    result=detector.predict(im)
    for d in result:
        assert d["category"] in detector.names
        assert .25 <= d["confidence"] <= 1
        x1,y1,x2,y2=d["box"]
        assert 0<=x1<x2<=im.width and 0<=y1<y2<=im.height


def test_tiled_detection_maps_to_original_coordinates():
    class FakeDetector(Detector):
        def __init__(self): pass
        def _predict(self,image,confidence):
            return [dict(category="milk",confidence=.9,box=[10,10,20,20])]
    result=FakeDetector().predict(Image.new("RGB",(800,500)),tiled=True)
    assert len(result) > 1
    assert any(d["box"][0]>300 for d in result)
    assert all(0<=d["box"][2]<=800 and 0<=d["box"][3]<=500 for d in result)
