# Submission checklist

Completed in the supplied project:

- [x] StockSense Pro scenario, problem definition and input/output scope.
- [x] Five categories, 100 images/category, balanced 70/15/15 split.
- [x] YOLO conversion, normalization, training augmentations.
- [x] Completed 30-epoch fine-tuning, weights and training curves.
- [x] Held-out precision, recall, mAP, confusion matrix and counting metrics.
- [x] Streamlit detection, counting, colored boxes, alerts, insights and exports.
- [x] Automated app/data/logic tests and lighting/clutter stress checks.
- [x] Source, dataset, notebook, model and detailed README.

Finish with your own details and accounts:

- [ ] Fill in student full name, registration number and school in README and submission_details.json.
- [ ] Upload the extracted project to your GitHub repository.
- [ ] Deploy on Streamlit Community Cloud and test the live link.
- [ ] Add both real URLs to README and submission_details.json.
- [ ] Provide the assessor access as instructed in the brief: ai.assignments@wacpinternational.org. Resolve the correct GitHub account if collaborator access is required; this package does not send an invitation.
- [ ] Obtain real user feedback and record it in FEEDBACK_FORM.md.
- [ ] Generate the final submission PDF and check all links.
- [ ] Confirm school requirements for acknowledging Codex assistance.

The included submission PDF is a draft. After updating the JSON:

```bash
python -m pip install reportlab==4.4.4
python scripts/make_submission.py
```

Upload the resulting `docs/StockSense_Submission.pdf` as the assignment document. Do not submit blank placeholders as a completed form.
