"""Streamlit entry point: streamlit run app.py"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from html import escape
import json
import time

import altair as alt
import pandas as pd
import streamlit as st

from stocksense.detector import Detector, annotate, png_bytes, read_image
from stocksense.logic import STATUS_COLORS, summarize

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="StockSense Pro · Shelf intelligence", page_icon="▥", layout="wide")
st.markdown("""<style>
.block-container {max-width:1450px;padding-top:1.5rem;padding-bottom:2rem;}
html, body, [class*="css"], [data-testid="stApp"] {font-family:Arial,Helvetica,sans-serif;}
.hero h1,.wordmark,h3 {font-family:Arial,Helvetica,sans-serif !important;}
div[data-testid="stMetric"] {background:white;border:1px solid #DEE3E9;border-radius:8px;padding:16px 20px;border-top:3px solid #25344A;}
div[data-testid="stMetricLabel"] {color:#63718A;}
div[data-testid="stMetricValue"] {font-weight:750;}
.eyebrow {color:#68788D;font-size:.7rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin-bottom:10px;}
.hero {padding:6px 0 22px;display:flex;align-items:center;justify-content:space-between;gap:20px;}
.hero h1 {font-size:2.2rem;letter-spacing:-.055em;margin:0;line-height:1.2;font-weight:750;}
.hero p {color:#63718A;font-size:.95rem;margin:10px 0 0;}
.hero-tag {font-size:.7rem;letter-spacing:.08em;border:1px solid #D2E5DF;color:#176A51;background:#EDF6F2;padding:8px 12px;border-radius:4px;white-space:nowrap;}
.wordmark {font-weight:800;font-size:1.28rem;letter-spacing:-.05em;color:#17243B;margin-bottom:5px;}
.wordmark small {font-size:.62rem;letter-spacing:.1em;background:#25344A;color:white;padding:3px 5px;vertical-align:middle;margin-left:5px;border-radius:3px;}
.priority-card {background:white;border:1px solid #DEE3E9;border-left:3px solid var(--status);padding:13px 15px;margin:8px 0;border-radius:5px;}
.priority-head {display:flex;justify-content:space-between;align-items:center;font-size:.9rem;gap:8px;}
.priority-head span {font-size:.72rem;font-weight:700;color:#526179;}
.priority-card p {font-size:.78rem;margin:5px 0 0;color:#63718A;}
.rules {display:flex;gap:10px;margin:14px 0;flex-wrap:wrap;}
.rule {font-size:.76rem;border:1px solid #DDE3EA;border-radius:5px;padding:7px 10px;background:white;}
.rule b {margin-right:5px;}
.stTabs [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #DDE3EA;}
.stTabs [data-baseweb="tab"] {font-weight:650;}
h3 {font-size:1.05rem !important;letter-spacing:-.02em;}
section[data-testid="stSidebar"] {border-right:1px solid #DDE3EA;}
@media (max-width:760px) {.hero-tag {display:none;} .hero h1 {font-size:1.8rem;}}
.note {color:#63718A;font-size:.85rem;}
/* These panels stay white in both themes, so their text must stay black. */
div[data-testid="stMetric"],
div[data-testid="stMetric"] *,
.priority-card,
.priority-card *,
.rule,
.rule * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
/* The sidebar follows the selected theme instead of using a fixed dark logo. */
.wordmark {color: inherit;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    return Detector(ROOT / "models/best.onnx", ROOT / "models/metadata.json")


@st.cache_data(max_entries=10, show_spinner=False)
def analyze(data, confidence, iou, tiled):
    image = read_image(data)
    start = time.perf_counter()
    detections = load_detector().predict(image, confidence, iou, tiled)
    return detections, round(time.perf_counter()-start, 2)


st.markdown('<div class="hero"><div><div class="eyebrow">StockSense Pro / Retail operations</div><h1>Shelf intelligence</h1><p>Review visible stock, inspect detections, and plan the next refill.</p></div><div class="hero-tag">GROCERY SHELF MONITOR</div></div>', unsafe_allow_html=True)

try:
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
except (OSError, ValueError):
    st.error("The model files are missing. Upload the complete models folder from the project ZIP.")
    st.stop()
classes = metadata["classes"]
with st.sidebar:
    st.markdown('<div class="wordmark">StockSense <small>PRO</small></div>',unsafe_allow_html=True)
    st.caption("RETAIL OPERATIONS WORKSPACE")
    st.divider()
    st.markdown("### Shelf settings")
    st.caption("Choose the categories this shelf is expected to carry.")
    expected = st.multiselect("Expected products", classes, default=classes, format_func=str.title)
    low = st.number_input("Low stock: at most", min_value=1, max_value=20, value=3, step=1)
    target = st.number_input("Target visible units per category", min_value=int(low)+1, max_value=100, value=max(6,int(low)+1), step=1)
    st.divider()
    with st.expander("Detection settings", expanded=False):
        confidence = st.slider("Minimum confidence", .05, .90, .25, .05)
        iou = st.slider("Duplicate-box overlap (IoU)", .10, .80, .45, .05)
        tiled = st.checkbox("Scan a wide shelf in overlapping sections", value=False,
                            help="Retains detail on larger images. Can create duplicate boxes at crop edges; inspect the result.")
        st.caption("Higher confidence removes uncertain detections, but may miss more products.")
    st.divider()
    st.markdown("**Five trained categories**")
    st.caption("Cereal · Coffee · Juice · Milk · Water")
    st.caption("Counts describe visible products in one image. Occluded products and backroom stock are not counted.")
    st.caption("Educational prototype · YOLO11n")

workspace, history_tab, model_tab = st.tabs(["Shelf scan", "Session history", "Model & evidence"])
with workspace:
    input_col, guide_col = st.columns([1.8,1], gap="large")
    with input_col:
        st.markdown("### Image workspace")
        source = st.radio("Image source", ["Example image", "Upload my image"], horizontal=True, label_visibility="collapsed")
        data, filename = None, None
        if source == "Example image":
            samples_path = ROOT / "samples/index.json"
            samples = json.loads(samples_path.read_text()) if samples_path.exists() else []
            if samples:
                choice = st.selectbox("Example", range(len(samples)), format_func=lambda i:samples[i]["title"])
                sample = samples[choice]
                filename = sample["file"]
                data = (ROOT / "samples" / filename).read_bytes()
                st.caption(sample["description"])
            else:
                st.info("Upload an image to get started.")
        else:
            uploaded = st.file_uploader("JPG, PNG, or WebP · up to 12 MB", type=["jpg","jpeg","png","webp"])
            if uploaded:
                data, filename = uploaded.getvalue(), uploaded.name
    with guide_col:
        st.markdown("### Stock rules")
        st.markdown(f'<div class="rules"><div class="rule"><b style="color:#CA3B3B">0</b> Out of stock</div><div class="rule"><b style="color:#B77711">1–{low}</b> Low stock</div><div class="rule"><b style="color:#09855D">{low+1}+</b> In stock</div></div>',unsafe_allow_html=True)
        st.caption("A zero is a restocking check, not proof of absence. Confirm the image covers the shelf and the model hasn’t missed an item.")
    image = None
    if data:
        try:
            image = read_image(data)
        except ValueError as exc:
            st.error(str(exc))
    signature = sha256(data + json.dumps([confidence,iou,tiled,expected,low,target]).encode()).hexdigest() if data else None
    requested = st.button("Analyze shelf", type="primary", disabled=image is None or not expected, use_container_width=True)
    demo_changed = source == "Example image" and image is not None and expected and st.session_state.get("result",{}).get("signature") != signature
    if requested or demo_changed:
        try:
            with st.spinner("Finding products and checking visible stock…"):
                detections, seconds = analyze(data, confidence, iou, tiled)
                visible = [d for d in detections if d["category"] in expected]
                rows = summarize(visible, expected, int(low), int(target))
                result = dict(signature=signature, filename=filename,
                              timestamp=datetime.now(timezone.utc).isoformat(),
                              detections=visible, rows=rows, seconds=seconds,
                              settings=dict(confidence=confidence,iou=iou,tiled=tiled,expected=expected,
                                            low_threshold=int(low),target=int(target)),
                              image_size=list(image.size), model="StockSense YOLO11n")
                st.session_state["result"] = result
                history = st.session_state.setdefault("scan_history", [])
                if not history or history[-1]["signature"] != signature:
                    history.append(result)
                    st.session_state["scan_history"] = history[-20:]
        except Exception as exc:
            st.error("The scan could not finish. Check that models/best.onnx and models/metadata.json were uploaded together.")
            with st.expander("Technical details"):
                st.code(str(exc))
    result = st.session_state.get("result")
    if result and result["signature"] == signature and image:
        rows = result["rows"]
        df = pd.DataFrame(rows)
        annotated = annotate(image, result["detections"], rows)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Visible products", int(df["count"].sum()))
        m2.metric("Out of stock", int((df["status"] == "Out of stock").sum()))
        m3.metric("Low stock", int((df["status"] == "Low stock").sum()))
        m4.metric("In stock", int((df["status"] == "In stock").sum()))
        st.caption(f"{len(expected)} expected categories · {result['seconds']:.2f}s inference · counts are image estimates")
        pictures, insights = st.columns([1.65, 1], gap="large")
        with pictures:
            st.markdown("### Visual shelf check")
            annotated_tab, original_tab = st.tabs(["Detected products", "Original image"])
            with annotated_tab:
                st.image(annotated, width="stretch")
                st.caption("Boxes show detected items. Their colors match the category’s stock status. Empty categories appear in the alert list.")
            with original_tab:
                st.image(image, width="stretch")
        with insights:
            st.markdown("### Restocking priorities")
            if not result["detections"]:
                st.warning("No expected products detected. Check the photo, selected categories, and confidence setting before restocking.")
            for row in rows:
                color=STATUS_COLORS[row["status"]]
                st.markdown(f'<div class="priority-card" style="--status:{color}"><div class="priority-head"><b>{escape(row["category"].title())}</b><span>{row["count"]} VISIBLE · {escape(row["status"].upper())}</span></div><p>{escape(row["action"])} · Top-up to target: {row["suggested_top_up"]}</p></div>',unsafe_allow_html=True)
            st.caption("Top-up quantities below use your chosen target. Sales performance cannot be inferred from a single shelf image.")
        st.markdown("### Category overview")
        table, chart = st.columns([1.2,1])
        with table:
            display = df[["category","count","status","suggested_top_up"]].rename(columns={"category":"Product","count":"Visible","status":"Status","suggested_top_up":"Top-up to target"})
            st.dataframe(display, hide_index=True, use_container_width=True)
        with chart:
            graph = alt.Chart(df).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("count:Q", title="Visible items", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("category:N", title=None, sort=classes),
                color=alt.Color("status:N",scale=alt.Scale(domain=list(STATUS_COLORS),range=list(STATUS_COLORS.values())),legend=None),
                tooltip=["category", "count", "status"]).properties(height=200)
            st.altair_chart(graph, use_container_width=True)
        with st.expander("Detection details"):
            details = [dict(product=d["category"],confidence=round(d["confidence"],3),x1=round(d["box"][0]),y1=round(d["box"][1]),x2=round(d["box"][2]),y2=round(d["box"][3])) for d in result["detections"]]
            st.dataframe(pd.DataFrame(details),hide_index=True,use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.download_button("Download stock CSV",df.to_csv(index=False).encode(),"stocksense_stock.csv","text/csv",use_container_width=True)
        c2.download_button("Download annotated image",png_bytes(annotated),"stocksense_annotated.png","image/png",use_container_width=True)
        c3.download_button("Download scan JSON",json.dumps(result,indent=2).encode(),"stocksense_scan.json","application/json",use_container_width=True)
    else:
        if result and image:
            st.info("Image or settings changed. Select Analyze shelf to update the results.")
        if image:
            st.image(image, caption="Ready to scan", width=560)
        if not expected:
            st.info("Select at least one expected product in the sidebar.")

with history_tab:
    st.markdown("### This session’s scans")
    st.caption("Last 20 scans. History resets when the session ends; repeated scans are not added together as inventory.")
    history = st.session_state.get("scan_history",[])
    if history:
        records = [dict(timestamp=r["timestamp"],image=r["filename"],visible_items=sum(x["count"] for x in r["rows"]),out_of_stock=sum(x["status"] == "Out of stock" for x in r["rows"]),low_stock=sum(x["status"] == "Low stock" for x in r["rows"])) for r in history]
        history_df = pd.DataFrame(records)
        st.dataframe(history_df,hide_index=True,use_container_width=True)
        st.download_button("Download session CSV",history_df.to_csv(index=False).encode(),"stocksense_session.csv","text/csv")
        if st.button("Clear session history"):
            st.session_state["scan_history"] = []
            st.rerun()
    else:
        st.info("Analyze an image to create your first entry.")

with model_tab:
    st.markdown("### What this model can do")
    st.write("A YOLO11n model fine-tuned on labeled grocery photographs detects five broad product categories. It does not identify brands, prices, expiry dates, or unsupported products.")
    st.json(metadata,expanded=False)
    metrics_file = ROOT / "reports/evaluation.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        st.markdown("### Held-out test results")
        a,b,c,d = st.columns(4)
        a.metric("Precision @ 0.25 confidence",f"{metrics['precision']:.1%}")
        b.metric("Recall @ 0.25 confidence",f"{metrics['recall']:.1%}")
        c.metric("Count MAE / category / image",f"{metrics['count_mae']:.2f}")
        d.metric("Exact five-category counts",f"{metrics['exact_count_rate']:.1%}")
        st.caption("IoU ≥ 0.5 for a correct detection. The metrics use the same ONNX engine as this app, with section scanning off.")
        st.dataframe(pd.DataFrame(metrics["per_class"]),hide_index=True,use_container_width=True)
        if metrics.get("stress_tests"):
            st.markdown("#### Lighting and clutter checks")
            st.dataframe(pd.DataFrame(metrics["stress_tests"]),hide_index=True,use_container_width=True)
        st.caption("This small test set estimates performance on this dataset. It does not establish reliability on new stores, unseen brands, or crowded full-resolution shelves.")
    st.markdown("### How a scan works")
    st.write("The image is resized with padding and normalized to 0–1. The model predicts boxes and category scores. Confidence filtering and duplicate suppression produce visible item counts, which are compared with the stock thresholds.")
    st.markdown("### Project evidence")
    st.write("The download includes the data split, training code, notebook, model weights, measured evaluation, and a submission checklist. No customer photographs are written to disk by the app; scan data is held temporarily in memory.")
    st.caption("Training source: Freiburg Groceries Dataset with the public Pascal VOC bounding-box extension. See README and DATASET_CARD for attribution and limitations.")
