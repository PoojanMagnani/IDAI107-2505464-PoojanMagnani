"""Expand the five included category archives before training or evaluation."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def ensure_dataset(root=ROOT):
    dataset=root/"dataset"
    if (dataset/"images/train").exists() and len(list((dataset/"images").rglob("*.png"))) == 500:
        return
    archives=sorted((dataset/"archives").glob("*.zip"))
    if not archives:
        raise FileNotFoundError("Dataset missing. Upload dataset/archives or run prepare_data.py first.")
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target=(dataset/entry.filename).resolve()
                if not target.is_relative_to(dataset.resolve()):
                    raise ValueError("Unsafe path inside dataset archive")
            z.extractall(dataset)


if __name__ == "__main__":
    ensure_dataset()
    print("Dataset ready: images, labels and original annotations extracted.")
