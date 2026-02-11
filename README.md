# Anant Neural Network Tool Box

Simple Streamlit app demonstrating a small neural network trained on the Pima Indians Diabetes dataset.

Quick start

1. Create a virtual environment and activate it.

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) Download dataset to `data/` (script included):

```bash
python scripts/download_data.py
```

4. Run the Streamlit app

```bash
streamlit run app.py
```

Files

- `requirements.txt` — Python dependencies
- `scripts/download_data.py` — download the CSV into `data/`
- `model_utils.py` — helper functions to save/load trained models
- `data/.gitkeep` and `models/.gitkeep` — placeholders so folders appear in git
- `.gitignore` — common ignores including model and data artifacts


