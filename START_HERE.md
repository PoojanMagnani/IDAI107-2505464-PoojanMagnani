# Upload to GitHub, then deploy to Streamlit

You do not need to train the model. Everything needed to run the app is included.

## 1. Extract the download

Unzip `StockSense_Pro_GitHub_Ready.zip`. Open its `stocksense-pro` folder. You should see `app.py`, `requirements.txt`, `models`, and the other folders.

Leave the five files inside `dataset/archives/` zipped. They contain the training data; they are not needed for app startup.

## 2. Create your GitHub repository

Suggested name: `IADAI201YOUR_ID-yourname`.

Upload the **files and folders inside** `stocksense-pro` to the repository root. The GitHub page should show `app.py` directly at the top level. Keep `models`, `stocksense`, `samples`, `reports`, `dataset`, `docs`, `scripts`, `notebooks`, and `tests` as folders.

Use “Add file → Upload files” and drag the contents into the upload area. If GitHub asks for smaller batches, upload folders in separate commits, preserving their names. GitHub Desktop can also publish the extracted folder.

Some file managers hide `.streamlit`. If that folder is missing on GitHub, create a file named `.streamlit/config.toml` in GitHub and paste the exact contents of `docs/streamlit_config_copy.txt`. This keeps the intended colors and upload limit.

Do not upload only the outer ZIP, move `best.onnx` out of `models`, or rename `app.py`.

## 3. Deploy

Open https://share.streamlit.io/ and connect the GitHub account that can access the repository.

- Create an app from the existing repository.
- Repository: your new repository.
- Branch: the branch containing your uploaded files (usually `main`).
- Main file path: `app.py`.
- Advanced settings: Python **3.12**.
- Deploy.

The first launch installs the listed dependencies. No secrets or API keys are needed. Cloud availability and resource limits are controlled by Streamlit.

## 4. Check the live app

The milk-shelf example should open with detection results. Try a second example, upload a photo, change expected categories and download a stock report. The supported categories are cereal, coffee, juice, milk and water.

Copy the live URL into README.md and `docs/submission_details.json`. Open the app in a private browser window to check accessibility before submitting.

## If something fails

| Symptom | Check |
|---|---|
| “Model files are missing” | `models/best.onnx` and `models/metadata.json` are present together |
| Import/dependency error | Python 3.12 is selected; root `requirements.txt` was uploaded unchanged |
| App file not found | `app.py` is at the repository root |
| Missing example | Upload the complete `samples/` folder |
| Different colors | Add `.streamlit/config.toml` from the included copy |
| Few/incorrect detections | Try a closer, sharper image; inspect boxes and confidence; unsupported products are outside this model's scope |

Official deployment reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
