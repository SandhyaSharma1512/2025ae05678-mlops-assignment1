"""Exploratory analysis charts and summary statistics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease.data import PROCESSED_PATH

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "eda"


def create_eda(data_path: Path = PROCESSED_PATH, output_dir: Path = OUT) -> dict:
    if not data_path.exists():
        raise FileNotFoundError("Processed CSV missing; run `python -m heart_disease.data --download`.")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    if "target" not in df:
        raise ValueError("Expected binary 'target' column in processed data.")

    df.describe(include="all").to_csv(output_dir / "summary_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(output_dir / "missing_values.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="target", ax=ax)
    ax.set_title("Heart disease target class balance")
    ax.set_xlabel("Target (0 = absent, 1 = present)")
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close(fig)

    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    cols = [c for c in numeric.columns if c != "target"][:12]
    if cols:
        df[cols].hist(figsize=(14, 10), bins=20)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_histograms.png", dpi=150)
        plt.close("all")
    return {"rows": int(len(df)), "columns": int(df.shape[1]), "target_balance": df["target"].value_counts().to_dict()}


if __name__ == "__main__":
    print(create_eda())
