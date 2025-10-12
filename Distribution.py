# distribution_top10.py
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== CONFIG ======
INPUT_CSV = "new_data_to_analysis.csv"
OUT_DIR = Path("dists")
TOP_N_FOR_HICARD = 10  

# ====== LOAD ======
df = pd.read_csv(INPUT_CSV)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Hàm vẽ biểu đồ ---
def plot_numeric_hist(series: pd.Series, bins: int = 30, title: str = "", fname: str = "hist.png"):
    plt.figure()
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    plt.hist(s, bins=bins)
    plt.title(title or f"Distribution of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()

def plot_categorical_bar(series: pd.Series, top_n: int = None, title: str = "", fname: str = "bar.png"):
    plt.figure()
    s = series.fillna("NA").astype(str)
    if top_n is not None:
        counts = s.value_counts().head(top_n)
        title_suffix = f" (top {top_n})"
    else:
        counts = s.value_counts()
        title_suffix = ""
    counts.plot(kind="bar")
    plt.title((title or f"Distribution of {series.name}") + title_suffix)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()

# --- Vẽ cho từng cột ---
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        plot_numeric_hist(df[col], bins=30, title=f"Distribution of {col}", fname=f"{col}_hist.png")
    else:
        if col.lower() in {"sku", "style", "core"}:
            top_n = 10                               
        else:
            top_n = None
        plot_categorical_bar(df[col], top_n=top_n, title=f"Distribution of {col}", fname=f"{col}_bar.png")

# --- Thêm log1p(Amount) để biểu đồ Amount đẹp hơn ---
if "Amount" in df.columns:
    df["log1p_Amount"] = np.log1p(df["Amount"])
    plot_numeric_hist(df["log1p_Amount"], bins=30,
                      title="Distribution of log1p(Amount)",
                      fname="log1p_Amount_hist.png")

print("✅ Saved charts to:", OUT_DIR.resolve().as_posix())
