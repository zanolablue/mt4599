"""
compare_clustering_baseline.py
================================
Script to compare clustering quality of:
  (A) Transformer encoder embeddings  (mean-pooled over window tokens)
  (B) Raw state vectors               (mean-pooled over window time-steps)

Uses the SAME PCA → k-means → silhouette / DB pipeline for both.
Optionally fits a GMM and reports BIC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare clustering quality: transformer embeddings vs raw state mean-pooling.\n"
            "Applies identical PCA → k-means → metrics pipeline to both representations."
        )
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings .npz from extract_embeddings.py (must contain 'emb_seq' or 'emb_last').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to windowed dataset .npz from build_window_dataset.py.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to use from the dataset (default: train).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Number of PCA components to retain (default: 32).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="3,4",
        help="Comma-separated list of K values for k-means (default: '3,4').",
    )
    parser.add_argument(
        "--gmm",
        action="store_true",
        default=True,
        help="Fit GMM and report BIC (default: True).",
    )
    parser.add_argument(
        "--no-gmm",
        dest="gmm",
        action="store_false",
        help="Disable GMM fitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for k-means and GMM (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/clustering_comparison",
        help="Directory for output files (default: results/clustering_comparison).",
    )
    return parser.parse_args()

---------------------------------------------------------------------------

def _load_embeddings(path: Path) -> np.ndarray:
    """
    Load transformer encoder embeddings from .npz.

    Prefers 'emb_last' (final-token CLS-style vector, shape (N, d_model)).
    Falls back to 'emb_seq' → mean-pools over the time axis → (N, d_model).
    """
    npz = np.load(path, allow_pickle=False)
    if "emb_last" in npz:
        emb = npz["emb_last"]
        print(f"  Loaded 'emb_last' embeddings: shape={emb.shape}")
    elif "emb_seq" in npz:
        emb_seq = npz["emb_seq"]          # (N, T, d_model)
        emb = emb_seq.mean(axis=1)        # mean-pool over tokens
        print(f"  Loaded 'emb_seq' → mean-pooled: shape={emb.shape}")
    else:
        raise KeyError(
            f"Embeddings file {path} must contain 'emb_last' or 'emb_seq'. "
            f"Found: {list(npz.files)}"
        )
    if not np.all(np.isfinite(emb)):
        raise ValueError("Non-finite values in transformer embeddings.")
    return emb.astype(np.float64)


def _load_raw_windows(dataset_path: Path, split: str) -> np.ndarray:
    """
    Load sliding-window arrays from dataset .npz and return the requested split.

    Returns X of shape (N, T, feature_dim).
    """
    npz = np.load(dataset_path, allow_pickle=False)
    key = f"X_{split}"
    if key not in npz:
        available = [k for k in npz.files if k.startswith("X_")]
        raise KeyError(f"Key '{key}' not found in dataset. Available: {available}")
    X = npz[key].astype(np.float64)   # (N, T, feature_dim)
    print(f"  Loaded raw windows '{key}': shape={X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite values in raw window data.")
    return X


def _mean_pool_windows(X: np.ndarray) -> np.ndarray:
    """
    Compute per-window mean over the time axis.

    Args:
        X: (N, T, D) windowed state array.
    Returns:
        x_raw: (N, D) mean-pooled feature vectors.
    """
    return X.mean(axis=1)   # (N, D)
--------------------------------------------------------------------------

def _fit_pca(features: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """Fit PCA and return projected features + fitted PCA object."""
    n_comp = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    projected = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {n_comp} components, {explained:.1%} variance explained")
    return projected, pca


def _run_kmeans(
    projected: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit k-means and return (labels, silhouette, davies_bouldin).
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(projected)

    # Need at least 2 non-trivial clusters for metric computation
    unique = np.unique(labels)
    if len(unique) < 2:
        return labels, float("nan"), float("nan")

    sil = silhouette_score(projected, labels)
    db = davies_bouldin_score(projected, labels)
    return labels, sil, db


def _run_gmm(
    projected: np.ndarray,
    k: int,
    seed: int,
) -> float:
    """Fit a diagonal-covariance GMM and return BIC."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=seed,
        n_init=3,
    )
    gmm.fit(projected)
    return float(gmm.bic(projected))


def _run_pipeline(
    features: np.ndarray,
    n_components: int,
    k_values: List[int],
    run_gmm: bool,
    seed: int,
    label: str,
) -> Tuple[Dict[str, Any], np.ndarray, PCA]:
    """
    Full pipeline for one representation.

    Returns:
        results: dict keyed by K with metric sub-dicts.
        projected: PCA-projected array (for plotting).
        pca: fitted PCA object.
    """
    print(f"\n--- Pipeline: {label} ---")
    projected, pca = _fit_pca(features, n_components)

    results: Dict[str, Any] = {}
    labels_k4: np.ndarray | None = None

    for k in k_values:
        print(f"    k-means k={k} …", end=" ", flush=True)
        labels, sil, db = _run_kmeans(projected, k, seed)
        entry: Dict[str, Any] = {"silhouette": sil, "db_index": db}

        if run_gmm:
            bic = _run_gmm(projected, k, seed)
            entry["gmm_bic"] = bic

        results[k] = entry
        print(f"sil={sil:.4f}  DB={db:.4f}" + (f"  BIC={bic:.1f}" if run_gmm else ""))

        if k == 4:
            labels_k4 = labels

    if labels_k4 is None:
        # Fall back to last k if 4 was not in k_values
        labels_k4 = labels 

    return results, projected, pca, labels_k4


_CMAP = "tab10"

def _plot_pca_clusters(
    projected: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """
    Scatter plot of first two PCA components coloured by cluster label.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    k = int(labels.max()) + 1
    cmap = plt.get_cmap(_CMAP)
    for c in range(k):
        mask = labels == c
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=6,
            alpha=0.55,
            color=cmap(c / max(k - 1, 1)),
            label=f"Cluster {c}",
            rasterized=True,
        )
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(markerscale=3, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def _print_table(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    k_values: List[int],
    run_gmm: bool,
) -> None:
    """Print a nicely formatted comparison table to stdout."""
    header_parts = ["Method               ", " K ", " Silhouette ", " DB Index "]
    if run_gmm:
        header_parts.append(" GMM BIC  ")
    header = "|".join(header_parts)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    method_display = {
        "transformer": "Transformer       ",
        "raw_state":   "Raw state (mean)  ",
    }
    for method_key in ["transformer", "raw_state"]:
        if method_key not in all_results:
            continue
        name = method_display[method_key]
        for k in k_values:
            m = all_results[method_key][k]
            sil = m["silhouette"]
            db = m["db_index"]
            row = f"{name} | {k} |   {sil:+.4f}   |  {db:.4f}  "
            if run_gmm:
                bic = m.get("gmm_bic", float("nan"))
                row += f"| {bic:10.1f}"
            print(row)
    print(sep + "\n")


def main() -> None:
    args = _parse_args()
    k_values = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading data …")
    emb_path = Path(args.embeddings)
    ds_path = Path(args.dataset)

    transformer_features = _load_embeddings(emb_path)
    X_windows = _load_raw_windows(ds_path, args.split)

    raw_features = _mean_pool_windows(X_windows)
    print(f"  Raw state mean-pool: shape={raw_features.shape}")

    n = min(len(transformer_features), len(raw_features))
    if len(transformer_features) != len(raw_features):
        print(
            f"  Warning: embedding count ({len(transformer_features)}) ≠ "
            f"window count ({len(raw_features)}). Truncating to {n}."
        )
    transformer_features = transformer_features[:n]
    raw_features = raw_features[:n]

    print("\n[2/4] Running PCA → k-means pipelines …")
    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    t_results, t_proj, _, t_labels_k4 = _run_pipeline(
        transformer_features,
        n_components=args.n_components,
        k_values=k_values,
        run_gmm=args.gmm,
        seed=args.seed,
        label="Transformer embeddings",
    )
    all_results["transformer"] = t_results

    r_results, r_proj, _, r_labels_k4 = _run_pipeline(
        raw_features,
        n_components=args.n_components,
        k_values=k_values,
        run_gmm=args.gmm,
        seed=args.seed,
        label="Raw state (mean-pooled)",
    )
    all_results["raw_state"] = r_results

    print("\n[3/4] Results summary:")
    _print_table(all_results, k_values, args.gmm)

    print("[4/4] Generating plot …")
    k_plot = 4 if 4 in k_values else k_values[-1]
    plot_path = output_dir / f"pca_raw_state_k{k_plot}.png"

    if k_plot != 4:
        _, r_labels_k4 = _run_kmeans(r_proj, k_plot, args.seed)[:2], None
        km_plot = KMeans(n_clusters=k_plot, random_state=args.seed, n_init=10)
        r_labels_kplot = km_plot.fit_predict(r_proj)
    else:
        r_labels_kplot = r_labels_k4

    _plot_pca_clusters(
        r_proj,
        r_labels_kplot,
        title=f"Raw state (mean-pool) — PCA projection (K={k_plot})",
        output_path=plot_path,
    )

    metrics_out: Dict[str, Any] = {
        "n_samples": n,
        "n_pca_components": args.n_components,
        "k_values": k_values,
        "split": args.split,
        "seed": args.seed,
        "results": {
            method: {
                str(k): scores for k, scores in method_results.items()
            }
            for method, method_results in all_results.items()
        },
    }
    metrics_path = output_dir / "clustering_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
