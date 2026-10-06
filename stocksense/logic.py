"""Pure stock logic, kept independent from the dashboard and model."""
from collections import Counter

STATUS_COLORS = {"Out of stock": "#EF4444", "Low stock": "#F59E0B", "In stock": "#10B981"}


def stock_status(count: int, low_threshold: int = 3) -> str:
    if count < 0 or low_threshold < 0:
        raise ValueError("Counts and thresholds must be non-negative.")
    if count == 0:
        return "Out of stock"
    return "Low stock" if count <= low_threshold else "In stock"


def summarize(detections: list[dict], expected: list[str], low_threshold: int = 3,
              target: int = 6) -> list[dict]:
    if target <= low_threshold:
        raise ValueError("Target must be greater than the low-stock threshold.")
    counts = Counter(d["category"] for d in detections)
    rows = []
    for category in dict.fromkeys(expected):
        count = counts[category]
        status = stock_status(count, low_threshold)
        scores = [d["confidence"] for d in detections if d["category"] == category]
        priority = {"Out of stock": 1, "Low stock": 2, "In stock": 3}[status]
        action = ("Check shelf; no items detected" if count == 0 else
                  "Restock needed" if status == "Low stock" else "Stock sufficient")
        rows.append(dict(category=category, count=count, status=status,
                         priority=priority, suggested_top_up=max(0, target-count),
                         mean_confidence=round(sum(scores)/len(scores), 4) if scores else None,
                         action=action))
    return sorted(rows, key=lambda r: (r["priority"], r["count"], r["category"]))
