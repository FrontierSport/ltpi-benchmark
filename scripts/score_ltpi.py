"""Stage 2 of the LTPI pipeline.

Convert a saved Tracklab state into flat tables, then score long-term player
identification with the Cost-Sensitive Identification Score (CSIS).

Pipeline:
    1. ``convert``  -- unpack ``sn-gamestate.pklz`` into ``main.parquet`` +
       ``embeddings.npy``.
    2. ``evaluate`` -- build a per-player train/test split, compute class
       centroids on the train part, classify test subtracks by cosine
       similarity and report CSIS over a set of rejection margins.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Configuration --------------------------------------------------------

FPS: int = 30
MIN_TRAIN_SECONDS: int = 60
MIN_TRAIN_FRAMES: int = MIN_TRAIN_SECONDS * FPS

# Ground-truth id encoding: gt_track_id = team_global * TEAM_ID_BASE + jersey.
TEAM_ID_BASE: int = 100

# CSIS cost weights.
COST_WRONG: float = 1.0
COST_UNKNOWN: float = 0.5

# Rejection margins to sweep (best CSIS is reported).
DEFAULT_MARGINS: tuple[float, ...] = (0.02,)

STATE_FILE: str = "sn-gamestate.pklz"
PARQUET_FILE: str = "main.parquet"
EMBEDDINGS_FILE: str = "embeddings.npy"

# Columns that are not needed downstream and may be missing.
DROP_COLUMNS: tuple[str, ...] = ("bbox_ltwh", "bbox_conf", "bbox_pitch")


@dataclass(frozen=True)
class CsisMetrics:
    """Cost-Sensitive Identification Score and its components."""

    csis: float
    coverage: float
    accuracy: float
    mis_id: float
    unk_rate: float
    n_total: int
    n_correct: int
    n_wrong: int
    n_unk: int


# --- Stage 2a: state -> flat tables --------------------------------------


def convert(results_dir: Path, video_id: str) -> None:
    """Unpack one video's dataframe from the Tracklab state.

    Embeddings are stored separately as ``embeddings.npy``; any remaining
    multi-dimensional array columns are dropped before writing the parquet.
    """
    states_file = results_dir / STATE_FILE
    member = f"{video_id}.pkl"

    with zipfile.ZipFile(states_file) as zf:
        click.echo(f"Files in state: {zf.namelist()}")
        with zf.open(member, force_zip64=True) as fp:
            df = pd.read_pickle(fp)

    click.echo(f"Read {member} ({len(df):,} rows)")

    if "embeddings" in df.columns:
        embeddings = np.stack(df["embeddings"].values)
        np.save(results_dir / EMBEDDINGS_FILE, embeddings)
        click.echo(f"Saved embeddings: {embeddings.shape}")
        df = df.drop(columns=["embeddings"])

    multidim_cols = [
        col
        for col in df.columns
        if df[col].dtype == object
        and isinstance(df[col].iloc[0], np.ndarray)
        and df[col].iloc[0].ndim > 1
    ]
    if multidim_cols:
        click.echo(f"Dropping multidim columns: {multidim_cols}")
        df = df.drop(columns=multidim_cols)

    df.to_parquet(results_dir / PARQUET_FILE, engine="pyarrow")
    click.echo(f"Wrote {PARQUET_FILE}")


# --- Stage 2b: train/test split ------------------------------------------


def _assign_split(group: pd.DataFrame) -> pd.DataFrame:
    """Label the first ``MIN_TRAIN_FRAMES`` of a player as train, rest as test."""
    group = group.copy()
    cumulative = group["duration"].cumsum().shift(fill_value=0)
    group["split"] = np.where(cumulative < MIN_TRAIN_FRAMES, "train", "test")
    return group


def build_split(fragments_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-subtrack train/test split table.

    Subtracks are ordered chronologically per (jersey_number, team); the
    earliest ones (up to one minute) form the enrollment/train set.
    """
    subtrack_stats = (
        fragments_df.groupby(
            ["jersey_number", "team", "subtrack_id"], as_index=False
        )
        .agg(min_frame=("image_id", "min"), max_frame=("image_id", "max"))
        .assign(duration=lambda x: x["max_frame"] - x["min_frame"])
        .sort_values(["jersey_number", "team", "min_frame"])
    )

    split_df = subtrack_stats.groupby(
        ["jersey_number", "team"], group_keys=False
    ).apply(_assign_split, include_groups=False)

    result = subtrack_stats[
        ["subtrack_id", "jersey_number", "team", "duration"]
    ].copy()
    result["split"] = split_df["split"].values
    return result


# --- Stage 2b: centroid model --------------------------------------------


def compute_centroids(
    df_train: pd.DataFrame,
) -> tuple[list[int], np.ndarray]:
    """Compute one mean-embedding centroid per ground-truth identity."""
    centroid_ids: list[int] = []
    centroids: list[np.ndarray] = []
    for gt_id in df_train["gt_track_id"].unique():
        subset = df_train[df_train["gt_track_id"] == gt_id]
        emb = np.stack(subset["embeddings"].values).squeeze()
        centroid_ids.append(int(gt_id))
        centroids.append(emb.mean(axis=0))
    return centroid_ids, np.stack(centroids)


def predict_subtracks(
    df: pd.DataFrame,
    test_subtracks: np.ndarray,
    centroid_ids: list[int],
    centroid_matrix: np.ndarray,
) -> pd.DataFrame:
    """Classify each test subtrack by cosine similarity to class centroids.

    The rejection ``margin`` is the gap between the top-1 and top-2 similarity.
    """
    predictions: list[dict[str, float]] = []
    for subtrack_id in tqdm(test_subtracks, desc="cosine"):
        subtrack_data = df[df["old_track_id"] == subtrack_id]
        if len(subtrack_data) == 0:
            continue

        emb = np.stack(subtrack_data["embeddings"].values).squeeze()
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        track_centroid = emb.mean(axis=0).reshape(1, -1)

        sims = cosine_similarity(track_centroid, centroid_matrix)[0]
        top2 = np.argsort(sims)[-2:][::-1]
        predictions.append(
            {
                "subtrack_id": subtrack_id,
                "true_gt": subtrack_data["gt_track_id"].iloc[0],
                "pred_gt": centroid_ids[top2[0]],
                "confidence": sims[top2[0]],
                "margin": sims[top2[0]] - sims[top2[1]],
            }
        )
    return pd.DataFrame(predictions)


# --- Stage 2b: metric ----------------------------------------------------


def compute_csis(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    is_unk: np.ndarray,
) -> CsisMetrics:
    """Compute the Cost-Sensitive Identification Score and its components."""
    n = len(true_labels)
    n_unk = int(is_unk.sum())
    n_accepted = n - n_unk

    accepted = ~is_unk
    n_correct = int(((true_labels == pred_labels) & accepted).sum())
    n_wrong = int(((true_labels != pred_labels) & accepted).sum())

    avg_cost = (n_wrong * COST_WRONG + n_unk * COST_UNKNOWN) / n
    return CsisMetrics(
        csis=1 - (avg_cost / COST_WRONG),
        coverage=n_accepted / n,
        accuracy=n_correct / n_accepted if n_accepted > 0 else 0.0,
        mis_id=n_wrong / n,
        unk_rate=n_unk / n,
        n_total=n,
        n_correct=n_correct,
        n_wrong=n_wrong,
        n_unk=n_unk,
    )


def sweep_margins(
    pred_df: pd.DataFrame, margins: tuple[float, ...]
) -> tuple[float, CsisMetrics]:
    """Evaluate CSIS over every margin and return the best (margin, metrics)."""
    header = (
        f"{'Margin':<8} {'CSIS':<8} {'Coverage':<10} "
        f"{'Accuracy':<10} {'MisID':<8} {'UNKRate':<8}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    best_margin = margins[0]
    best_metrics: CsisMetrics | None = None
    for margin in margins:
        is_unk = pred_df["margin"].values < margin
        metrics = compute_csis(
            pred_df["true_gt"].values, pred_df["pred_gt"].values, is_unk
        )
        click.echo(
            f"{margin:<8.3f} {metrics.csis:<8.3f} {metrics.coverage:<10.3f} "
            f"{metrics.accuracy:<10.3f} {metrics.mis_id:<8.3f} "
            f"{metrics.unk_rate:<8.3f}"
        )
        if best_metrics is None or metrics.csis > best_metrics.csis:
            best_margin, best_metrics = margin, metrics

    assert best_metrics is not None
    return best_margin, best_metrics


def evaluate(
    results_dir: Path,
    ds_path: Path,
    video_id: str,
    split: str,
    margins: tuple[float, ...],
) -> None:
    """Score long-term identification for one video and print CSIS."""
    df = pd.read_parquet(results_dir / PARQUET_FILE)
    df["embeddings"] = list(np.load(results_dir / EMBEDDINGS_FILE))
    df = df.drop(columns=list(DROP_COLUMNS), errors="ignore")
    df["old_track_id"] = df["track_id"]

    video_dir = ds_path / split / video_id
    fragments_df = pd.read_csv(video_dir / "subtracks.csv")

    split_table = build_split(fragments_df)
    train_subtracks = set(
        split_table.loc[split_table["split"] == "train", "subtrack_id"]
    )
    test_subtracks = split_table.loc[
        split_table["split"] == "test", "subtrack_id"
    ].values

    df_train = df[df["old_track_id"].isin(train_subtracks)].copy()
    click.echo(
        f"Train subtracks: {len(train_subtracks)} "
        f"({len(df_train)} samples, "
        f"{df_train['gt_track_id'].nunique()} classes); "
        f"test subtracks: {len(test_subtracks)}"
    )

    centroid_ids, centroid_matrix = compute_centroids(df_train)
    click.echo(f"Prepared {len(centroid_ids)} class centroids")

    pred_df = predict_subtracks(
        df, test_subtracks, centroid_ids, centroid_matrix
    )
    if pred_df.empty:
        click.echo("WARNING: no predictions made")
        return

    best_margin, best_metrics = sweep_margins(pred_df, margins)
    click.echo(
        f"\nBest margin: {best_margin} (CSIS={best_metrics.csis:.3f})"
    )


# --- CLI ------------------------------------------------------------------


@click.command()
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with the saved Tracklab state (sn-gamestate.pklz).",
)
@click.option("--video-id", default='2', help="Video id to evaluate.")
@click.option(
    "--ds-path",
    default='data/SoccerNetGS',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root of the LTPI dataset.",
)
@click.option(
    "--split",
    default='test',
    type=click.Choice(["train", "valid", "test", "challenge"]),
    help="Dataset split the video belongs to.",
)
@click.option(
    "--margin",
    "margins",
    multiple=True,
    type=float,
    help="Rejection margin(s) to sweep. Repeatable; best CSIS is reported.",
)
def main(
    results_dir: Path,
    video_id: str,
    ds_path: Path,
    split: str,
    margins: tuple[float, ...],
) -> None:
    """Convert a Tracklab state and score long-term player identification."""
    convert(results_dir, video_id)
    evaluate(
        results_dir,
        ds_path,
        video_id,
        split,
        margins or DEFAULT_MARGINS,
    )


if __name__ == "__main__":
    main()
