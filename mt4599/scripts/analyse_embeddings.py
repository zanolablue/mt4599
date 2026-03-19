"""
analyse_embeddings.py
=====================
Statistical latent-state analysis of transformer encoder embeddings.

Pipeline (follows thesis Sections 5.4 – 5.7):
  1. Load embeddings + dataset (emb_last, normalisation stats, split labels)
  2. PCA  — scree plot, 2-D scatter coloured by split
  3. K-means — elbow / silhouette over K=2..8, final K=4 clustering
  4. GMM    — BIC selection over K=2..8, soft assignments
  5. HMM    — fitted on per-window k-means labels, transition matrix heatmap
  6. Behavioural interpretation — cluster centroid kinematics in original units
  7. All figures saved to --output-dir as high-resolution PNGs

Usage:
    python -m mt4599.scripts.analyse_embeddings \
        --embeddings runs/transformer_baseline/embeddings_all.npz \
        --dataset    data/processed/euroc_W128_S16_dataset.npz \
        --output-dir runs/transformer_baseline/analysis \
        --n-clusters 4 \
        --pca-components 32
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on compute nodes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=ConvergenceWarning if False else UserWarning)

# ── colour palette (one per cluster, up to 8) ────────────────────────────────
CLUSTER_COLOURS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
                   "#F4A261", "#264653", "#A8DADC", "#6D6875"]

FEATURE_NAMES = ["p_x","p_y","p_z",
                 "v_x","v_y","v_z",
                 "q_x","q_y","q_z","q_w",
                 "w_x","w_y","w_z",
                 "a_x","a_y","a_z"]

BEHAVIOUR_LABELS = {
    # filled in heuristically after centroid inspection (Section 5.7.2)
    # keys are cluster indices; values are tentative labels
}


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent-state analysis of transformer embeddings.")
    p.add_argument("--embeddings",    required=True,  help="embeddings_all.npz")
    p.add_argument("--dataset",       required=True,  help="windowed dataset .npz (for mu/sigma)")
    p.add_argument("--output-dir",    required=True,  help="Directory for saved figures and CSVs")
    p.add_argument("--n-clusters",    type=int, default=4,
                   help="Number of clusters K for final k-means / GMM (default: 4)")
    p.add_argument("--pca-components",type=int, default=32,
                   help="PCA dimensionality before clustering (default: 32)")
    p.add_argument("--hmm-n-iter",    type=int, default=200,
                   help="EM iterations for HMM fitting (default: 200)")
    p.add_argument("--max-k",         type=int, default=8,
                   help="Maximum K to search for elbow/BIC (default: 8)")
    return p.parse_args()


def _load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return emb_last (N,128), split_labels (N,), meta dict."""
    npz  = np.load(path, allow_pickle=False)
    emb  = npz["emb_last"].astype(np.float32)           # (N, d_model)
    labs = npz["split_labels"].astype(str)               # (N,)  'train'/'val'/'test'
    meta = json.loads(str(npz["meta_json"]))
    return emb, labs, meta


def _load_normalisation(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return mu (16,) and sigma (16,) from the windowed dataset."""
    npz = np.load(dataset_path, allow_pickle=False)
    return npz["mu"].astype(np.float64), npz["sigma"].astype(np.float64)


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. PCA
# ═══════════════════════════════════════════════════════════════════════════

def run_pca(
    emb: np.ndarray,
    n_components: int,
    out_dir: Path,
    split_labels: np.ndarray,
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on all embeddings, save scree plot and 2-D scatter.
    Returns projected embeddings (N, n_components) and fitted PCA object.
    """
    print("\n[1] PCA")
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)

    pca_full = PCA(n_components=min(emb.shape[1], 64)).fit(emb_s)
    pca      = PCA(n_components=n_components).fit(emb_s)
    proj     = pca.transform(emb_s)   # (N, n_components)

    explained      = pca_full.explained_variance_ratio_
    cumulative     = np.cumsum(explained)
    var_retained   = float(pca.explained_variance_ratio_.sum())
    print(f"    PCA({n_components}) retains {var_retained*100:.1f}% variance")

    # ── scree plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(explained)+1), explained * 100,
                color="#457B9D", alpha=0.8)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot")
    axes[0].set_xlim(0.5, min(32, len(explained)) + 0.5)

    axes[1].plot(range(1, len(cumulative)+1), cumulative * 100,
                 "o-", color="#E63946", markersize=4)
    axes[1].axhline(90, ls="--", color="gray", lw=1, label="90%")
    axes[1].axhline(95, ls=":",  color="gray", lw=1, label="95%")
    axes[1].axvline(n_components, ls="--", color="#2A9D8F", lw=1.5,
                    label=f"d={n_components} ({var_retained*100:.0f}%)")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0.5, min(64, len(cumulative)) + 0.5)
    axes[1].set_ylim(0, 102)
    fig.suptitle("PCA of Transformer Encoder Embeddings", fontsize=13)
    _savefig(fig, out_dir / "pca_scree.png")

    # ── 2-D scatter coloured by split ────────────────────────────────────
    split_colour = {"train": "#457B9D", "val": "#2A9D8F", "test": "#E63946", "": "#999"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in ["train", "val", "test"]:
        mask = split_labels == sp
        if mask.sum() == 0:
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   s=4, alpha=0.4, color=split_colour[sp], label=sp, rasterized=True)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Embedding Space — PC1 vs PC2 (coloured by split)")
    ax.legend(markerscale=4, fontsize=9)
    _savefig(fig, out_dir / "pca_scatter_split.png")

    return proj, pca


# ═══════════════════════════════════════════════════════════════════════════
# 2. K-means — elbow + silhouette, then final clustering
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans(
    proj: np.ndarray,
    n_clusters: int,
    max_k: int,
    out_dir: Path,
) -> np.ndarray:
    """
    Search K=2..max_k, plot elbow and silhouette, return labels for n_clusters.
    """
    print("\n[2] K-means")
    inertias, sil_scores, ks = [], [], range(2, max_k + 1)

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbs = km.fit_predict(proj)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(proj, lbs, sample_size=min(5000, len(lbs))))
        print(f"    k={k}  inertia={km.inertia_:.1f}  silhouette={sil_scores[-1]:.4f}")

    # ── elbow + silhouette ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(ks), inertias, "o-", color="#457B9D")
    axes[0].axvline(n_clusters, ls="--", color="#E63946", lw=1.5,
                    label=f"chosen K={n_clusters}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("K-means Elbow Curve"); axes[0].legend()

    axes[1].plot(list(ks), sil_scores, "o-", color="#2A9D8F")
    axes[1].axvline(n_clusters, ls="--", color="#E63946", lw=1.5,
                    label=f"chosen K={n_clusters}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores"); axes[1].legend()
    fig.suptitle("K-means Cluster Selection", fontsize=13)
    _savefig(fig, out_dir / "kmeans_selection.png")

    # ── final clustering ──────────────────────────────────────────────────
    km_final = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels   = km_final.fit_predict(proj)
    print(f"    Final k-means K={n_clusters}: "
          f"silhouette={silhouette_score(proj, labels, sample_size=min(5000,len(labels))):.4f}")

    # ── 2-D scatter coloured by cluster ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   s=4, alpha=0.4, color=CLUSTER_COLOURS[k],
                   label=f"Cluster {k}", rasterized=True)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"K-means Clustering (K={n_clusters}) — PC1 vs PC2")
    ax.legend(markerscale=4, fontsize=9)
    _savefig(fig, out_dir / "kmeans_scatter.png")

    # ── cluster size bar chart ────────────────────────────────────────────
    counts = np.bincount(labels, minlength=n_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"C{k}" for k in range(n_clusters)], counts,
           color=CLUSTER_COLOURS[:n_clusters])
    ax.set_ylabel("Window Count"); ax.set_title("K-means Cluster Sizes")
    for i, c in enumerate(counts):
        ax.text(i, c + 10, str(c), ha="center", fontsize=9)
    _savefig(fig, out_dir / "kmeans_sizes.png")

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# 3. GMM — BIC selection, soft assignments, uncertainty map
# ═══════════════════════════════════════════════════════════════════════════

def run_gmm(
    proj: np.ndarray,
    n_clusters: int,
    max_k: int,
    out_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BIC selection over K=2..max_k, fit final GMM, return hard labels and
    soft responsibilities (N, K).
    """
    print("\n[3] GMM")
    bics, ks = [], range(2, max_k + 1)
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, random_state=42, max_iter=300)
        gm.fit(proj)
        bics.append(gm.bic(proj))
        print(f"    k={k}  BIC={bics[-1]:.1f}")

    # ── BIC curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), bics, "o-", color="#457B9D")
    ax.axvline(n_clusters, ls="--", color="#E63946", lw=1.5,
               label=f"chosen K={n_clusters}")
    ax.set_xlabel("K"); ax.set_ylabel("BIC")
    ax.set_title("GMM BIC Curve (lower = better)"); ax.legend()
    _savefig(fig, out_dir / "gmm_bic.png")

    # ── final GMM ─────────────────────────────────────────────────────────
    gm_final = GaussianMixture(n_components=n_clusters, covariance_type="full",
                               n_init=10, random_state=42, max_iter=500)
    gm_final.fit(proj)
    resp   = gm_final.predict_proba(proj)     # (N, K) soft assignments
    labels = gm_final.predict(proj)           # (N,)   hard labels

    # ── scatter with uncertainty (max responsibility as alpha) ────────────
    max_resp = resp.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in range(n_clusters):
        mask = labels == k
        axes[0].scatter(proj[mask, 0], proj[mask, 1],
                        s=4, alpha=0.4, color=CLUSTER_COLOURS[k],
                        label=f"Cluster {k}", rasterized=True)
    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
    axes[0].set_title(f"GMM Hard Assignments (K={n_clusters})")
    axes[0].legend(markerscale=4, fontsize=9)

    sc = axes[1].scatter(proj[:, 0], proj[:, 1],
                         s=4, c=max_resp, cmap="viridis",
                         alpha=0.5, rasterized=True, vmin=0.5, vmax=1.0)
    plt.colorbar(sc, ax=axes[1], label="Max Responsibility γ_k")
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
    axes[1].set_title("GMM Assignment Certainty")
    fig.suptitle("Gaussian Mixture Model — Latent Space", fontsize=13)
    _savefig(fig, out_dir / "gmm_scatter.png")

    # ── average uncertainty per cluster ───────────────────────────────────
    entropy = -np.sum(resp * np.log(resp + 1e-12), axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [entropy[labels == k].mean() for k in range(n_clusters)]
    ax.bar([f"C{k}" for k in range(n_clusters)], means,
           color=CLUSTER_COLOURS[:n_clusters])
    ax.set_ylabel("Mean Assignment Entropy"); ax.set_title("GMM Uncertainty per Cluster")
    _savefig(fig, out_dir / "gmm_uncertainty.png")

    return labels, resp


# ═══════════════════════════════════════════════════════════════════════════
# 4. HMM — fitted on sequence of k-means labels
# ═══════════════════════════════════════════════════════════════════════════

def run_hmm(
    labels: np.ndarray,
    n_clusters: int,
    n_iter: int,
    out_dir: Path,
) -> None:
    """
    Fit a categorical HMM on the sequence of cluster labels.
    Saves transition matrix heatmap and stationary distribution bar chart.

    Note: labels here are assumed to come from windows in temporal order
    (they do — extract_embeddings preserves order within each split).
    """
    print("\n[4] HMM")
    obs = labels.reshape(-1, 1)

    model = hmm.CategoricalHMM(n_components=n_clusters, n_iter=n_iter,
                                random_state=42, verbose=False)
    model.fit(obs)

    A     = model.transmat_          # (K, K) transition matrix
    pi    = model.startprob_         # (K,)   initial distribution

    # Stationary distribution via eigenvector
    eigvals, eigvecs = np.linalg.eig(A.T)
    stat = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    stat = np.abs(stat) / np.abs(stat).sum()

    print("    HMM transition matrix:")
    for i in range(n_clusters):
        row = "  ".join(f"{A[i,j]:.3f}" for j in range(n_clusters))
        print(f"      C{i}: [{row}]  (self-persist={A[i,i]:.3f})")
    print(f"    Stationary dist: {np.round(stat, 3)}")

    # ── transition matrix heatmap ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(A, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks(range(n_clusters))
    axes[0].set_yticks(range(n_clusters))
    axes[0].set_xticklabels([f"C{k}" for k in range(n_clusters)])
    axes[0].set_yticklabels([f"C{k}" for k in range(n_clusters)])
    axes[0].set_xlabel("To State"); axes[0].set_ylabel("From State")
    axes[0].set_title("HMM Transition Matrix P")
    for i in range(n_clusters):
        for j in range(n_clusters):
            axes[0].text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if A[i,j] > 0.5 else "black")

    # ── stationary distribution ───────────────────────────────────────────
    axes[1].bar([f"C{k}" for k in range(n_clusters)], stat,
                color=CLUSTER_COLOURS[:n_clusters])
    axes[1].set_ylabel("Stationary Probability")
    axes[1].set_title("HMM Stationary Distribution π∞")
    for i, s in enumerate(stat):
        axes[1].text(i, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)
    fig.suptitle("Hidden Markov Model — Regime Transition Structure", fontsize=13)
    _savefig(fig, out_dir / "hmm_transitions.png")

    # ── persistence bar (diagonal of A) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"C{k}" for k in range(n_clusters)],
           [A[k, k] for k in range(n_clusters)],
           color=CLUSTER_COLOURS[:n_clusters])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Self-Transition Probability P_ii")
    ax.set_title("Regime Persistence (HMM Diagonal)")
    for k in range(n_clusters):
        ax.text(k, A[k,k] + 0.01, f"{A[k,k]:.3f}", ha="center", fontsize=9)
    _savefig(fig, out_dir / "hmm_persistence.png")

    np.save(out_dir / "hmm_transmat.npy", A)
    np.save(out_dir / "hmm_stationary.npy", stat)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Behavioural interpretation — cluster centroid kinematics
# ═══════════════════════════════════════════════════════════════════════════

def run_behavioural_interpretation(
    emb_last: np.ndarray,       # (N, d_model) — raw (pre-PCA) embeddings
    labels: np.ndarray,         # (N,) k-means labels
    dataset_path: Path,
    n_clusters: int,
    out_dir: Path,
) -> None:
    """
    Map cluster labels back to the original 16-D state space via the
    normalisation inverse transform.  Plot per-cluster mean ± std for the
    kinematic channels that drive behavioural interpretation (Section 5.7.2):
        velocity magnitude  ||v||
        accel magnitude     ||a||
        angular rate        ||ω||

    Also saves a radar / profile plot of all 16 feature means per cluster.
    """
    print("\n[5] Behavioural interpretation")
    mu, sigma = _load_normalisation(dataset_path)

    # We don't have direct access to per-window raw states here, but we
    # can use the dataset X arrays to compute cluster-mean kinematics.
    # Load X_train (normalised) and reconstruct unnormalised states.
    npz      = np.load(dataset_path, allow_pickle=False)
    X_train  = npz["X_train"]   # (N_train, W, 16)
    X_val    = npz["X_val"]
    X_test   = npz["X_test"]
    X_all    = np.concatenate([X_train, X_val, X_test], axis=0)  # (N, W, 16)

    # Use the last timestep of each window as the representative state
    # (consistent with emb_last being the final-token embedding)
    s_norm   = X_all[:, -1, :]                    # (N, 16)  normalised
    s_raw    = s_norm * sigma[None, :] + mu[None, :]  # (N, 16)  original units

    # Velocity / accel / angular rate magnitudes (behavioural proxies)
    v_mag = np.linalg.norm(s_raw[:, 3:6],  axis=1)   # ||v||
    a_mag = np.linalg.norm(s_raw[:, 13:16], axis=1)   # ||a||
    w_mag = np.linalg.norm(s_raw[:, 10:13], axis=1)   # ||ω||

    # ── per-cluster magnitude box plots ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, mag, title, unit in zip(
        axes,
        [v_mag,      a_mag,        w_mag],
        ["||v|| (m/s)", "||a|| (m/s²)", "||ω|| (rad/s)"],
        ["m/s",     "m/s²",       "rad/s"],
    ):
        data = [mag[labels == k] for k in range(n_clusters)]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="white", lw=2))
        for patch, col in zip(bp["boxes"], CLUSTER_COLOURS[:n_clusters]):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        ax.set_xticklabels([f"C{k}" for k in range(n_clusters)])
        ax.set_ylabel(unit); ax.set_title(title)
    fig.suptitle("Cluster Kinematics — Behavioural Proxies", fontsize=13)
    _savefig(fig, out_dir / "behaviour_magnitudes.png")

    # ── per-cluster feature mean heatmap (all 16 features, normalised) ────
    cluster_means_norm = np.array(
        [s_norm[labels == k].mean(axis=0) for k in range(n_clusters)]
    )   # (K, 16)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(cluster_means_norm, cmap="RdBu_r", aspect="auto",
                   vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="Normalised mean")
    ax.set_xticks(range(16))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {k}" for k in range(n_clusters)])
    ax.set_title("Cluster Mean State Vectors (normalised units)")
    _savefig(fig, out_dir / "behaviour_feature_heatmap.png")

    # ── summary table ─────────────────────────────────────────────────────
    print("\n    Cluster kinematic summary (mean ± std):")
    print(f"    {'Cluster':>8}  {'||v|| m/s':>12}  {'||a|| m/s²':>12}  {'||ω|| rad/s':>12}  {'N':>6}")
    rows = []
    for k in range(n_clusters):
        mask = labels == k
        row  = dict(
            cluster=k,
            v_mean=float(v_mag[mask].mean()),  v_std=float(v_mag[mask].std()),
            a_mean=float(a_mag[mask].mean()),  a_std=float(a_mag[mask].std()),
            w_mean=float(w_mag[mask].mean()),  w_std=float(w_mag[mask].std()),
            n=int(mask.sum()),
        )
        rows.append(row)
        print(f"    C{k:>6}    "
              f"{row['v_mean']:6.3f}±{row['v_std']:.3f}   "
              f"{row['a_mean']:6.3f}±{row['a_std']:.3f}   "
              f"{row['w_mean']:6.3f}±{row['w_std']:.3f}   "
              f"{row['n']:>6}")

    with (out_dir / "cluster_kinematics.json").open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"  saved → {out_dir / 'cluster_kinematics.json'}")

    # ── tentative behavioural labels (heuristic) ──────────────────────────
    # Sort clusters by velocity magnitude to assign behavioural labels
    v_means  = [v_mag[labels == k].mean() for k in range(n_clusters)]
    a_means  = [a_mag[labels == k].mean() for k in range(n_clusters)]
    w_means  = [w_mag[labels == k].mean() for k in range(n_clusters)]
    order    = np.argsort(v_means)   # low → high velocity

    tentative = {}
    if n_clusters == 4:
        tentative[int(order[0])] = "Hover / Steady State"
        tentative[int(order[1])] = "Low-speed Inspection"
        tentative[int(order[2])] = "Moderate Motion"
        tentative[int(order[3])] = "Aggressive Manoeuvre"
    else:
        for rank, k in enumerate(order):
            tentative[int(k)] = f"Regime {rank+1} (v̄={v_means[k]:.2f} m/s)"

    print("\n    Tentative behavioural labels (velocity-ranked):")
    for k, label in sorted(tentative.items()):
        print(f"      C{k}: {label}")

    with (out_dir / "tentative_labels.json").open("w") as f:
        json.dump(tentative, f, indent=2)

    # ── labelled scatter (PC1 vs PC2) with tentative behaviour names ──────
    # Need proj — reload from saved file
    proj_path = out_dir / "_proj_cache.npy"
    if proj_path.exists():
        proj = np.load(proj_path)
        fig, ax = plt.subplots(figsize=(9, 6))
        for k in range(n_clusters):
            mask  = labels == k
            label = tentative.get(k, f"C{k}")
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       s=4, alpha=0.35, color=CLUSTER_COLOURS[k],
                       label=f"C{k}: {label}", rasterized=True)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title("Latent Space — Tentative Behavioural Regimes")
        ax.legend(markerscale=4, fontsize=8, loc="upper right")
        _savefig(fig, out_dir / "behaviour_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Latent-State Analysis of Transformer Embeddings")
    print("=" * 60)

    emb, split_labels, meta = _load_embeddings(Path(args.embeddings))
    print(f"\nLoaded embeddings: {emb.shape}  splits: "
          f"train={np.sum(split_labels=='train')}, "
          f"val={np.sum(split_labels=='val')}, "
          f"test={np.sum(split_labels=='test')}")

    K   = args.n_clusters
    dpc = args.pca_components

    # 1. PCA
    proj, pca = run_pca(emb, dpc, out_dir, split_labels)
    np.save(out_dir / "_proj_cache.npy", proj)   # used by behavioural plot

    # 2. K-means
    km_labels = run_kmeans(proj, K, args.max_k, out_dir)
    np.save(out_dir / "kmeans_labels.npy", km_labels)

    # 3. GMM
    gmm_labels, gmm_resp = run_gmm(proj, K, args.max_k, out_dir)
    np.save(out_dir / "gmm_labels.npy",   gmm_labels)
    np.save(out_dir / "gmm_resp.npy",     gmm_resp)

    # 4. HMM (on k-means labels — hard assignments, temporal order preserved)
    run_hmm(km_labels, K, args.hmm_n_iter, out_dir)

    # 5. Behavioural interpretation
    run_behavioural_interpretation(
        emb_last=emb,
        labels=km_labels,
        dataset_path=Path(args.dataset),
        n_clusters=K,
        out_dir=out_dir,
    )

    # ── save run config ───────────────────────────────────────────────────
    cfg = vars(args)
    cfg["n_embeddings"] = int(emb.shape[0])
    cfg["d_model"]      = int(emb.shape[1])
    with (out_dir / "analysis_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Analysis complete.  All outputs in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
