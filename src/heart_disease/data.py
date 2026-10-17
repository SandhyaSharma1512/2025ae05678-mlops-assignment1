"""Dataset acquisition and preparation utilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_disease.csv"
TARGET_CANDIDATES = ("num", "target", "condition", "diagnosis")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI feature names and map the original target to binary."""
    frame = df.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    target_col = next((c for c in TARGET_CANDIDATES if c in frame.columns), None)
    if target_col is None:
        # The ucimlrepo data object can expose the target separately; the downloader
        # below combines it into a column named 'num'. This branch helps manual CSV use.
        raise ValueError(
            f"Could not identify target column. Expected one of {TARGET_CANDIDATES}; "
            f"found {list(frame.columns)}"
        )
    frame = frame.rename(columns={target_col: "target"})
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame = frame.dropna(subset=["target"])
    frame["target"] = (frame["target"] > 0).astype(int)
    frame = frame.replace("?", pd.NA)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="ignore")
    return frame


def download_dataset(output_path: Path = RAW_PATH) -> Path:
    """Fetch UCI Heart Disease data through ucimlrepo and save a combined CSV."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one target column, got {list(targets.columns)}")
    target = targets.iloc[:, 0].rename("num")
    combined = features.join(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path


def prepare_dataset(raw_path: Path = RAW_PATH, output_path: Path = PROCESSED_PATH) -> Path:
    """Clean data and write processed CSV, preserving missing values for pipeline imputation."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run `python -m heart_disease.data --download`."
        )
    raw = pd.read_csv(raw_path)
    cleaned = normalize_columns(raw)
    cleaned = cleaned.drop_duplicates()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download dataset from UCI")
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-path", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()
    if args.download:
        download_dataset(args.raw_path)
    result = prepare_dataset(args.raw_path, args.output_path)
    print(f"Prepared dataset: {result}")


if __name__ == "__main__":
    main()
