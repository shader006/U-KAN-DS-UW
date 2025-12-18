"""
Plot scalar metrics from the exported TensorBoard CSVs for version_0, version_1, and version_2.

Usage (from repo root):
    py -3 plot_metrics.py \
        --outdir lightning_logs_archive\\plots

You can also pass custom CSVs:
    py -3 plot_metrics.py --files lightning_logs_archive\\version_0\\U-net.csv lightning_logs_archive\\version_1\\U-net+DS.csv lightning_logs_archive\\version_2\\U-net+DS+UW.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import matplotlib.pyplot as plt
import pandas as pd


def default_files(repo_root: Path) -> List[Path]:
    """Return the three default CSV paths."""
    return [
        repo_root / "lightning_logs_archive" / "version_0" / "U-net.csv",
        repo_root / "lightning_logs_archive" / "version_1" / "U-net+DS.csv",
        repo_root / "lightning_logs_archive" / "version_2" / "U-net+DS+UW.csv",
    ]


def load_data(paths: Sequence[Path]) -> Dict[str, pd.DataFrame]:
    """Load each CSV into a DataFrame keyed by a readable label."""
    data = {}
    for path in paths:
        label = path.stem  # e.g., U-net, U-net+DS, U-net+DS+UW
        df = pd.read_csv(path)
        data[label] = df
    return data


def sanitize_tag(tag: str) -> str:
    """Make a filesystem-friendly tag name for saving plots."""
    return tag.replace("/", "_")


def add_epoch_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'epoch' column to each tag row using the logged epoch tag."""
    epoch_df = (
        df[df["tag"] == "epoch"][["step", "value"]]
        .sort_values("step")
        .rename(columns={"value": "epoch"})
    )
    if epoch_df.empty:
        return df.assign(epoch=pd.NA)

    merged = []
    for tag in df["tag"].unique():
        if tag == "epoch":
            merged.append(df[df["tag"] == "epoch"].assign(epoch=df[df["tag"] == "epoch"]["value"]))
            continue
        tag_df = df[df["tag"] == tag].sort_values("step")
        tag_with_epoch = pd.merge_asof(
            tag_df,
            epoch_df,
            on="step",
            direction="backward",
        )
        merged.append(tag_with_epoch)
    return pd.concat(merged, ignore_index=True)


def plot_separate(data: Dict[str, pd.DataFrame], tags: Iterable[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for label, df in data.items():
        df = add_epoch_column(df)
        version_dir = outdir / label
        version_dir.mkdir(parents=True, exist_ok=True)
        for tag in tags:
            tag_df = df[df["tag"] == tag].dropna(subset=["epoch"])
            if tag_df.empty:
                continue
            plt.figure()
            plt.plot(tag_df["epoch"], tag_df["value"], marker="o")
            plt.title(f"{label} - {tag}")
            plt.xlabel("epoch")
            plt.ylabel("value")
            plt.grid(True, alpha=0.3)
            safe_name = sanitize_tag(tag)
            outfile = version_dir / f"{safe_name}.png"
            plt.tight_layout()
            plt.savefig(outfile, dpi=150)
            plt.close()
            print(f"Saved {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot scalar metrics from TensorBoard CSV exports."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        help="CSV files to plot. Defaults to the three version_* CSVs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("lightning_logs_archive") / "plots",
        help="Directory to write plots (PNG).",
    )
    parser.add_argument(
        "--include-tags",
        nargs="+",
        help="Optional list of tags to plot. Default: all except 'epoch'.",
    )

    args = parser.parse_args()
    repo_root = Path(".").resolve()
    files = args.files or default_files(repo_root)

    data = load_data(files)
    all_tags: Set[str] = set()
    for df in data.values():
        all_tags.update(df["tag"].unique())
    tags = (
        args.include_tags
        if args.include_tags
        else sorted(t for t in all_tags if t != "epoch")
    )

    plot_separate(data, tags, args.outdir)


if __name__ == "__main__":
    main()
