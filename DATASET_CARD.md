# Dataset card

**Original:** Freiburg Groceries Dataset, Philipp Jund, Nichola Abdo, Andreas Eitel, Wolfram Burgard (2016). https://arxiv.org/abs/1611.05799 · https://github.com/PhilJd/freiburg_groceries_dataset

**Detection annotations:** https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset — Pascal VOC extension of the original dataset. Its README credits the original authors and describes 4,947 images at 224 × 224 with product boxes.

**Included subset:** 100 unique decoded-pixel images each for cereal, coffee, juice, milk and water. Seed 42; 70/15/15 per class. This is a custom educational subset, not the official Freiburg benchmark split.

**Files:** `dataset/archives/<category>.zip` contains the selected PNGs, converted YOLO labels and original XML annotations for that category. The manifest and split summary are outside the archives. The preparation script reproduces the selection from the source repository. `reports/provenance.json` identifies the source commit and model hash.

**Label conventions:** Existing source boxes were used, not manually drawn anew. Coordinates are clamped to the image bounds, invalid boxes rejected, then normalized. Pixel normalization for model input is separate from box-coordinate normalization.

**Known quality limitations:** A visual inspection found background products can be left unannotated in crowded shelf photographs. For example, the wide-shelf example contains many more visible packages than its annotation count. Metrics therefore measure agreement with these supplied annotations, not definitive whole-shelf inventory accuracy. Some images are close views of only one or two products. The five categories do not cover every grocery product and the subset contains no dedicated negative/background-only training set. Similar packaging or scenes can cross splits despite exact duplicate removal. Capture sessions/store IDs were not available for group splitting.

**Rights:** All credit and rights to third-party photographs and original annotations remain with their creators. No explicit standalone license file was found in the detection-extension repository; this project does not invent one or relicense those files under its software license. The subset is included with provenance for the requested educational experiment. Consult the original authors' terms before broader redistribution or commercial use.

**Recommended next collection:** Obtain consent for new store photographs, annotate every visible supported product consistently, include unknown/empty shelves, and split by store/capture session before augmenting. Keep an independently audited final test set.
