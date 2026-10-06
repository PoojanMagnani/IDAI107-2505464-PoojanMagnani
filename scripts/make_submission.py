"""Create the assignment PDF from real details and measured project results.

Requires: python -m pip install reportlab==4.4.4
Fill docs/submission_details.json first; blank fields produce a clearly marked draft.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    info=json.loads((ROOT/"docs/submission_details.json").read_text())
    metrics=json.loads((ROOT/"reports/evaluation.json").read_text())
    standard=json.loads((ROOT/"reports/ultralytics_test_metrics.json").read_text())
    draft=any(not str(v).strip() for v in info.values())
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom",fontName="Helvetica",fontSize=10,leading=15,textColor=colors.HexColor("#334155"),spaceAfter=10))
    styles["Title"].textColor=colors.HexColor("#17243B")
    styles["Heading2"].textColor=colors.HexColor("#087F72")
    story=[]
    def paragraph(text,style="BodyCustom"):
        story.append(Paragraph(text,styles[style]))
    def table(rows,widths=None):
        data=[[Paragraph(escape(str(x)),styles["BodyCustom"]) for x in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EDF4F3")),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#D7DFE7")),("INNERGRID",(0,0),(-1,-1),.25,colors.HexColor("#E5E7EB")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9)]))
        story.append(t);story.append(Spacer(1,14))
    paragraph("StockSense Pro","Title")
    paragraph("AI-based retail shelf intelligence","Heading2")
    paragraph("DRAFT TEMPLATE — student details and live links still required." if draft else "Project submission — Scenario 2")
    labels={"student_name":"Student's full name","candidate_registration_number":"Candidate registration number","school_name":"School name","crs_name":"CRS name","course_name":"Course name","github_url":"GitHub repository","streamlit_url":"Live Streamlit application"}
    table([["Submission field","Details"]]+[[labels[k],v or "[TO BE COMPLETED]"] for k,v in info.items()],[170,340])
    for key,label in [("github_url","GitHub repository"),("streamlit_url","Streamlit app")]:
        if info[key].startswith("https://"):
            paragraph(f'<link href="{escape(info[key])}" color="#087F72">Open {label}</link>')
    paragraph("Problem and solution","Heading2")
    paragraph("Manual shelf checks can miss low stock. This project detects cereal, coffee, juice, milk and water in an uploaded photograph, counts visible items, and converts counts into threshold-based stock alerts. It identifies broad categories, not brands or complete store inventory.")
    paragraph("Implementation","Heading2")
    paragraph("A balanced 500-image subset of the Freiburg Groceries detection extension was split into 350 training, 75 validation and 75 test images. Each category has 100 images. Source Pascal VOC boxes were converted to YOLO format; exact pixel duplicates were excluded before splitting. Augmentations affect only training data.")
    paragraph("YOLO11n was initialized from COCO weights and fine-tuned for 30 epochs at 224 x 224, batch 16, AdamW learning rate 0.001, seed 42, with the first 10 layers frozen. Best validation weights were exported to ONNX for lightweight CPU deployment. The Streamlit app includes annotations, stock metrics, priority alerts, exports and session history.")
    story.append(PageBreak())
    paragraph("Measured evaluation","Heading2")
    table([[f"Metric — {metrics['images']} held-out test images","Result"],["App precision at confidence 0.25",f"{metrics['precision']:.2%}"],["App recall at confidence 0.25",f"{metrics['recall']:.2%}"],["F1",f"{metrics['f1']:.2%}"],["Count MAE / category / image",f"{metrics['count_mae']:.4f}"],["Exact five-category count rate",f"{metrics['exact_count_rate']:.2%}"],["Ultralytics test mAP@0.50",f"{standard['metrics/mAP50(B)']:.2%}"],["Ultralytics test mAP@0.50:0.95",f"{standard['metrics/mAP50-95(B)']:.2%}"]],[345,165])
    paragraph("App detection metrics use a confidence threshold of 0.25, class-aware NMS at 0.45, and one-to-one matches at IoU >= 0.50. Standard Ultralytics mAP is a separate evaluator. The test set was not used for choosing the best epoch. Full metrics and per-image counts are included in reports/.")
    paragraph("Stock decisions","Heading2")
    paragraph("0 detected items: out of stock; 1-3: low stock; 4 or more: in stock. This resolves the overlapping count-3 ranges in the assignment. Users select expected categories and adjust thresholds/targets. Zero detections prompt a shelf check rather than proving absence.")
    paragraph("Testing and limitations","Heading2")
    paragraph("Tests cover threshold boundaries, missing categories, invalid images, normalization, suppression, model inference, split integrity and Streamlit controls. A low-light test reduced exact count agreement from 48% to 40%; the cluttered subset reached 36.36%. Small or obscured products, similar packaging and incomplete background annotations remain limitations. The model is an educational prototype; its metrics do not establish accuracy on new stores. Human feedback and live-cloud testing must be recorded after deployment.")
    paragraph("References and assistance","Heading2")
    for text in ["Jund et al. (2016), The Freiburg Groceries Dataset. https://arxiv.org/abs/1611.05799","Detection annotation extension: https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset","Redmon et al. (2016), You Only Look Once. https://arxiv.org/abs/1506.02640","Ultralytics YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/","Developed with Codex assistance. Review the code and disclose assistance according to school requirements. Source dataset and model credits are preserved in the repository."]:
        paragraph(escape(text))
    screenshot=ROOT/"docs/dashboard.png"
    if screenshot.exists():
        story.append(PageBreak());paragraph("Application screenshot","Heading2")
        from PIL import Image as PILImage
        w,h=PILImage.open(screenshot).size
        scale=min(510/w,660/h)
        story.append(Image(str(screenshot),width=w*scale,height=h*scale))
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(42,25,"StockSense Pro | CRS Artificial Intelligence"+(" | DRAFT" if draft else ""))
        canvas.drawRightString(A4[0]-42,25,str(doc.page))
    path=ROOT/"docs/StockSense_Submission.pdf"
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)


if __name__ == "__main__":
    main()
