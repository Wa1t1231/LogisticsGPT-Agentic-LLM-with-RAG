"""Generate quick charts from pandas DataFrame via matplotlib."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import uuid

def plot_dataframe(df: pd.DataFrame, x: str, y: str, title: str | None = None) -> Path:
    """Return path to PNG file containing a simple line chart."""
    fig, ax = plt.subplots()
    df.plot(x=x, y=y, ax=ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    out_path = Path(f"/mnt/data/plot_{uuid.uuid4().hex}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
