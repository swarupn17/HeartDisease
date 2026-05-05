# Heart Sound Analyzer

## Setup

1. Make sure you are using Python 3.12.4 (TensorFlow is not available for Python 3.14 in this setup).

If conda base is active, run:

```bash
conda deactivate
```

2. Create and activate a virtual environment with the pyenv Python:

```bash
/Users/swarup/.pyenv/versions/3.12.4/bin/python -m venv .venv
source .venv/bin/activate
```

3. Confirm interpreter version:

```bash
python --version
```

Expected: `Python 3.12.4`

4. Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

5. Run the app from the project root:

```bash
.venv/bin/python app.py
```

6. Open `http://127.0.0.1:5000` in your browser.

## Notes

- The model files must be present locally in `models/` for the app to run, but they should not be committed to GitHub.
- Uploaded audio is saved temporarily as `temp.wav`.
- The Grad-CAM output is written to `static/output.png`.

## GitHub upload safety

- Do not commit `.env`; use `.env.example` as the template.
- Keep large local artifacts out of the repository: `dataset/`, `models/`, `frontend/node_modules/`, `frontend/dist/`, and generated notebook outputs.
- If you need to share model weights, publish them separately or use Git LFS instead of storing them directly in the main repo.
- Rotate any API keys that were placed in `.env` before pushing this project publicly.