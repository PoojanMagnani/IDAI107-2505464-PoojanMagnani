from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from unpack_dataset import ensure_dataset


def pytest_sessionstart(session):
    ensure_dataset()
