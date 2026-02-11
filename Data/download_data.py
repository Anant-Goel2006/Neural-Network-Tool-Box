"""Download the Pima Indians Diabetes dataset into a local data/ folder."""
from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def download(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {dest}...")
    urllib.request.urlretrieve(URL, dest)
    print("Done.")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "pima-indians-diabetes.csv"
    download(out)
