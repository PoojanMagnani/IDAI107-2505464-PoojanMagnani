import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_balanced_split_and_no_exact_duplicate_leakage():
    rows=list(csv.DictReader((ROOT/"dataset/manifest.csv").open()))
    assert len(rows) == 500
    assert len({r["pixel_sha256"] for r in rows}) == 500
    for name in ["cereal","coffee","juice","milk","water"]:
        for split,count in [("train",70),("val",15),("test",15)]:
            assert sum(r["category"]==name and r["split"]==split for r in rows) == count


def test_yolo_annotations_are_valid():
    for path in (ROOT/"dataset/labels").rglob("*.txt"):
        assert (ROOT/"dataset/images"/path.parent.name/(path.stem+".png")).exists()
        for row in path.read_text().splitlines():
            c,x,y,w,h=map(float,row.split())
            assert c in range(5)
            assert 0<w<=1 and 0<h<=1
            assert x-w/2 >= -1e-6 and x+w/2 <= 1+1e-6
            assert y-h/2 >= -1e-6 and y+h/2 <= 1+1e-6
