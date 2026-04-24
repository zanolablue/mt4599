"""
analyse_embeddings.py  —  v5 (semantic cluster remapping for stable colours)
===========================================================================
Full statistical latent-state analysis for MT4599 thesis.

Sections produced (matching thesis Chapter 7):
  7.2  PCA embedding structure
  7.3  K-means cluster selection + final clustering
  7.4  GMM soft assignments + uncertainty
  7.5  HMM transition structure + dwell times
  7.6  Behavioural interpretation (kinematics + transition graph)
  7.7  Sequence-level behavioural profiles
  App  Random baseline ablation

Primary analysis: emb_mean, K=4
Secondary:        emb_last, K=3 (robustness check)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import geom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
# Colourblind-safe palette (Okabe & Ito, 2008).
# Colours are pinned to SEMANTIC roles, not array indices, so they can never
# be accidentally swapped when the cluster ordering changes.
#
# K=4 semantic assignment (after canonicalise_labels_by_behaviour):
#   C0  Hover / Low Motion            → dark navy   #003f5c
#   C1  Steady Translating Flight     → steel blue  #58508d  (was sky-blue)
#   C2  Orientation-Dominant Manouvre → light blue  #bc5090  (was teal/green)
#   C3  Aggressive / High-Speed       → gold        #ffa600
#
# All four are distinguishable under deuteranopia, protanopia, and tritanopia.
# None of the combinations include a red–green pair.

# Semantic colour lookup — keys are the canonical cluster IDs (0-3 for K=4).
_SEMANTIC_COLOURS_K4 = {
    0: "#003f5c",   # C0: Hover / Low Motion          — dark navy
    1: "#58508d",   # C1: Steady Translating Flight    — slate purple
    2: "#bc5090",   # C2: Orientation-Dominant         — mauve/orchid
    3: "#ffa600",   # C3: Aggressive / High-Speed      — amber/gold
}

_SEMANTIC_COLOURS_K3 = {
    0: "#003f5c",   # C0: Active Flight                — dark navy
    1: "#58508d",   # C1: Low-Speed / Hover            — slate purple
    2: "#bc5090",   # C2: Orientation-Dominant         — mauve/orchid
}

# Fallback for any K not explicitly listed above.
_FALLBACK_COLOURS = ["#003f5c", "#58508d", "#bc5090", "#ffa600",
                     "#ff6361", "#2f4b7c", "#665191", "#a05195"]

# ── diagnostic / non-cluster colour aliases ────────────────────────────────
# Used in scree plots, elbow curves, PCA split scatters, etc.
# Completely separate from cluster colours so they never interfere.
_DIAG_A = "#1b7fcc"   # medium blue   (was #457B9D / #E63946 red)
_DIAG_B = "#ffa600"   # amber         (was #2A9D8F teal-green)
_DIAG_C = "#58508d"   # slate purple  (was #E9C46A yellow — hard to read)
_DIAG_VLINE = "#cc4400"  # burnt orange for vertical reference lines (not red)
_DIAG_TRAIN = "#1b7fcc"
_DIAG_VAL   = "#bc5090"
_DIAG_TEST  = "#ffa600"


def cluster_colour_map(n_clusters: int) -> Dict[int, str]:
    """Return a dict mapping canonical cluster ID -> hex colour.

    Colours are pinned to semantic roles via the lookup tables above, so
    the mapping is stable across every plot regardless of how K-means
    internally numbers the clusters (which is resolved *before* this
    function is ever called, by canonicalise_labels_by_behaviour).
    """
    if n_clusters == 4:
        return dict(_SEMANTIC_COLOURS_K4)
    if n_clusters == 3:
        return dict(_SEMANTIC_COLOURS_K3)
    return {k: _FALLBACK_COLOURS[k % len(_FALLBACK_COLOURS)]
            for k in range(n_clusters)}


# Behavioural labels assigned from kinematic evidence.
# These assume cluster IDs have already been remapped onto semantic roles.
BEHAVIOUR_LABELS_K4 = {
    0: "Hover / Low Motion",
    1: "Steady Translating Flight",
    2: "Orientation-Dominant Manoeuvre",
    3: "Aggressive / High-Speed Motion",
}
BEHAVIOUR_LABELS_K3 = {
    0: "Active Flight",
    1: "Low-Speed / Hover",
    2: "Orientation-Dominant",
}

FEATURE_NAMES = [
    "Δp_x (m, world)", "Δp_y (m, world)", "Δp_z (m, world)",
    "v_x (m/s, world)", "v_y (m/s, world)", "v_z (m/s, world)",
    "q_x (-)", "q_y (-)", "q_z (-)", "q_w (-)",
    "ω_x (rad/s, body)", "ω_y (rad/s, body)", "ω_z (rad/s, body)",
    "a_x (m/s², world)", "a_y (m/s², world)", "a_z (m/s², world)",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def remap_labels(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(x)] for x in labels], dtype=int)


def canonicalise_labels_by_centroid(
    proj: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Give clusters a stable ordering based on their centroid in PCA space.
    Sort by PC1, then PC2.
    """
    centroids = np.array([
        proj[labels == k].mean(axis=0) for k in range(n_clusters)
    ])
    order = sorted(range(n_clusters), key=lambda k: (centroids[k, 0], centroids[k, 1]))
    mapping = {old_k: new_k for new_k, old_k in enumerate(order)}
    return remap_labels(labels, mapping), mapping


def canonicalise_labels_by_behaviour(
    labels: np.ndarray,
    n_clusters: int,
    dataset_path: Path,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Dict[str, float]]]:
    """
    Remap raw cluster IDs onto semantic behavioural roles using simple
    kinematic summaries instead of geometric cluster order.

    K=4:
      C0 Hover / Low Motion          -> lowest translational speed
      C3 Aggressive / High-Speed     -> highest translational speed
      C2 Orientation-Dominant        -> highest angular speed among remainder
      C1 Steady Translating Flight   -> remaining cluster

    K=3:
      C1 Low-Speed / Hover           -> lowest translational speed
      C2 Orientation-Dominant        -> highest angular speed among remainder
      C0 Active Flight               -> remaining cluster
    """
    X_all, mu, sigma, _ = _load_states(dataset_path)
    s_norm = X_all[:, -1, :]
    s_raw = s_norm * sigma[None, :] + mu[None, :]
    dp_mag = np.linalg.norm(s_raw[:, 0:3], axis=1)
    v_mag = np.linalg.norm(s_raw[:, 3:6], axis=1)
    w_mag = np.linalg.norm(s_raw[:, 10:13], axis=1)
    ac_mag = np.linalg.norm(s_raw[:, 13:16], axis=1)

    stats = {
        k: {
            "dp_mag": float(dp_mag[labels == k].mean()),
            "v_mag": float(v_mag[labels == k].mean()),
            "w_mag": float(w_mag[labels == k].mean()),
            "ac_mag": float(ac_mag[labels == k].mean()),
        }
        for k in range(n_clusters)
    }

    if n_clusters == 4:
        remaining = set(range(n_clusters))

        hover = min(
            remaining,
            key=lambda k: (stats[k]["v_mag"], stats[k]["w_mag"], stats[k]["dp_mag"]),
        )
        remaining.remove(hover)

        aggressive = max(
            remaining,
            key=lambda k: (stats[k]["v_mag"], stats[k]["dp_mag"], stats[k]["ac_mag"]),
        )
        remaining.remove(aggressive)

        orientation = max(
            remaining,
            key=lambda k: (stats[k]["w_mag"], -stats[k]["v_mag"]),
        )
        remaining.remove(orientation)

        steady = remaining.pop()
        mapping = {
            hover: 0,
            steady: 1,
            orientation: 2,
            aggressive: 3,
        }
        return remap_labels(labels, mapping), mapping, stats

    if n_clusters == 3:
        remaining = set(range(n_clusters))

        hover = min(
            remaining,
            key=lambda k: (stats[k]["v_mag"], stats[k]["w_mag"], stats[k]["dp_mag"]),
        )
        remaining.remove(hover)

        orientation = max(
            remaining,
            key=lambda k: (stats[k]["w_mag"], -stats[k]["v_mag"]),
        )
        remaining.remove(orientation)

        active = remaining.pop()
        mapping = {
            active: 0,
            hover: 1,
            orientation: 2,
        }
        return remap_labels(labels, mapping), mapping, stats

    canonical_labels, mapping = canonicalise_labels_by_centroid(
        np.column_stack([v_mag, w_mag]),
        labels,
        n_clusters,
    )
    return canonical_labels, mapping, stats


def align_labels_to_reference(
    ref_labels: np.ndarray,
    other_labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Align another clustering to a reference clustering by maximising overlap.
    Useful for GMM vs KMeans on the same samples.
    """
    conf = np.zeros((n_clusters, n_clusters), dtype=int)
    for r, o in zip(ref_labels, other_labels):
        conf[int(o), int(r)] += 1

    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {old_k: new_k for old_k, new_k in zip(row_ind, col_ind)}
    return remap_labels(other_labels, mapping), mapping


# EuRoC difficulty from sequence name
def _difficulty(name: str) -> str:
    n = name.lower()
    if "difficult" in n:
        return "difficult"
    if "medium" in n:
        return "medium"
    return "easy"


def _environment(name: str) -> str:
    return "MH" if name.upper().startswith("MH") else "Vicon"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--pca-components", type=int, default=32)
    p.add_argument("--hmm-n-iter", type=int, default=300)
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Bootstrap resamples for CI on silhouette (default 50)",
    )
    return p.parse_args()


def _savefig(fig, path: Path, dpi=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def _load_embeddings(path: Path):
    npz = np.load(path, allow_pickle=False)
    meta = json.loads(str(npz["meta_json"]))
    emb_last = npz["emb_last"].astype(np.float32)
    emb_mean = npz["emb_mean"].astype(np.float32)
    split_labels = npz["split_labels"].astype(str)
    seq_origin = npz["seq_origin"].astype(str) if "seq_origin" in npz else None
    return emb_last, emb_mean, split_labels, seq_origin, meta


def _load_states(dataset_path: Path):
    npz = np.load(dataset_path, allow_pickle=False)
    X_all = np.concatenate([npz["X_train"], npz["X_val"], npz["X_test"]], axis=0)
    mu, sigma = npz["mu"].astype(np.float64), npz["sigma"].astype(np.float64)

    seq_labels = None
    if "seq_labels_train" in npz:
        seq_labels = np.concatenate([
            npz["seq_labels_train"].astype(str),
            npz["seq_labels_val"].astype(str),
            npz["seq_labels_test"].astype(str),
        ])
    return X_all, mu, sigma, seq_labels


# ═══════════════════════════════════════════════════════════════════════════
# 7.2  PCA
# ═══════════════════════════════════════════════════════════════════════════

def run_pca(emb, n_components, split_labels, tag, out_dir):
    print(f"\n[PCA — {tag}]")
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)
    pca_full = PCA(n_components=min(emb.shape[1], 64)).fit(emb_s)
    pca = PCA(n_components=n_components).fit(emb_s)
    proj = pca.transform(emb_s)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA({n_components}) retains {var*100:.1f}% variance")

    expl = pca_full.explained_variance_ratio_
    cum = np.cumsum(expl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        range(1, min(33, len(expl) + 1)),
        expl[:32] * 100,
        color=_DIAG_A,
        alpha=0.85,
    )
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot")

    axes[1].plot(range(1, len(cum) + 1), cum * 100, "o-", color=_DIAG_VLINE, ms=4)
    axes[1].axhline(90, ls="--", color="gray", lw=1, label="90%")
    axes[1].axhline(95, ls=":", color="gray", lw=1, label="95%")
    axes[1].axvline(
        n_components,
        ls="--",
        color=_DIAG_B,
        lw=1.5,
        label=f"d={n_components} ({var*100:.0f}%)",
    )
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative (%)")
    axes[1].set_title("Cumulative Variance")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0.5, min(64, len(cum)) + 0.5)
    axes[1].set_ylim(0, 102)
    fig.suptitle(f"PCA of Encoder Embeddings [{tag}]", fontsize=13)
    _savefig(fig, out_dir / f"pca_scree_{tag}.png")

    colours = {"train": _DIAG_TRAIN, "val": _DIAG_VAL, "test": _DIAG_TEST}
    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in ["train", "val", "test"]:
        mask = split_labels == sp
        if not mask.any():
            continue
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=4,
            alpha=0.4,
            color=colours[sp],
            label=sp,
            rasterized=True,
        )
    ax.set_xlabel("PC1 (largest variance direction in embedding space)")
    ax.set_ylabel("PC2 (second-largest variance direction)")
    ax.set_title(f"Embedding Space — PC1 vs PC2 [{tag}]")
    ax.legend(markerscale=4, fontsize=9)
    _savefig(fig, out_dir / f"pca_scatter_split_{tag}.png")

    return proj, pca


# ═══════════════════════════════════════════════════════════════════════════
# 7.3  K-means
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans(proj, n_clusters, max_k, n_bootstrap, tag, out_dir, dataset_path):
    print(f"\n[K-means — {tag}  K={n_clusters}]")
    inertias, sils, dbs, ks = [], [], [], range(2, max_k + 1)

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbs = km.fit_predict(proj)
        inertias.append(km.inertia_)
        sil = silhouette_score(proj, lbs, sample_size=min(5000, len(lbs)))
        sils.append(sil)
        dbs.append(davies_bouldin_score(proj, lbs))
        print(f"  k={k}  inertia={km.inertia_:.0f}  sil={sil:.4f}  DB={dbs[-1]:.4f}")

    # ── selection plots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(list(ks), inertias, "o-", color=_DIAG_A)
    axes[0].axvline(n_clusters, ls="--", color=_DIAG_VLINE, lw=1.5, label=f"K={n_clusters}")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Curve")
    axes[0].legend()

    axes[1].plot(list(ks), sils, "o-", color=_DIAG_B)
    axes[1].axvline(n_clusters, ls="--", color=_DIAG_VLINE, lw=1.5, label=f"K={n_clusters}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores")
    axes[1].legend()

    axes[2].plot(list(ks), dbs, "o-", color=_DIAG_C)
    axes[2].axvline(n_clusters, ls="--", color=_DIAG_VLINE, lw=1.5, label=f"K={n_clusters}")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].set_title("Davies-Bouldin (lower=better)")
    axes[2].legend()
    fig.suptitle(f"K-means Cluster Selection [{tag}]", fontsize=13)
    _savefig(fig, out_dir / f"kmeans_selection_{tag}.png")

    # ── bootstrap confidence interval on silhouette ───────────────────────
    print(f"  Bootstrap CI on silhouette (K={n_clusters}, n={n_bootstrap})...")
    km_final = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km_final.fit_predict(proj)

    # Remap cluster IDs onto semantic behavioural roles once so colours and
    # labels stay consistent across every downstream plot.
    labels, mapping, proxy_stats = canonicalise_labels_by_behaviour(
        labels,
        n_clusters,
        dataset_path,
    )
    palette = cluster_colour_map(n_clusters)
    label_lookup = BEHAVIOUR_LABELS_K4 if n_clusters == 4 else BEHAVIOUR_LABELS_K3

    print("  Behavioural relabelling:")
    for old_k, new_k in sorted(mapping.items(), key=lambda kv: kv[1]):
        stats = proxy_stats[old_k]
        print(
            f"    raw C{old_k} -> C{new_k}"
            f" [{label_lookup.get(new_k, f'C{new_k}')}]"
            f"  v={stats['v_mag']:.3f}  w={stats['w_mag']:.3f}"
            f"  dp={stats['dp_mag']:.4f}  ac={stats['ac_mag']:.2f}"
        )

    boot_sils = []
    rng = np.random.default_rng(42)
    N = len(proj)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        lbs_b = labels[idx]
        if len(np.unique(lbs_b)) < 2:
            continue
        boot_sils.append(
            silhouette_score(proj[idx], lbs_b, sample_size=min(3000, N))
        )
    boot_sils = np.array(boot_sils)
    ci_lo, ci_hi = np.percentile(boot_sils, [2.5, 97.5])
    final_sil = silhouette_score(proj, labels, sample_size=min(5000, N))
    print(f"  Silhouette = {final_sil:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── scatter ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=4,
            alpha=0.4,
            color=palette[k],
            label=f"C{k}",
            rasterized=True,
        )
    ax.set_xlabel("PC1 (largest variance direction in embedding space)")
    ax.set_ylabel("PC2 (second-largest variance direction)")
    ax.set_title(f"K-means (K={n_clusters}) — PC1 vs PC2 [{tag}]")
    ax.legend(markerscale=4, fontsize=9)
    _savefig(fig, out_dir / f"kmeans_scatter_{tag}_K{n_clusters}.png")

    # ── cluster sizes ─────────────────────────────────────────────────────
    counts = np.bincount(labels, minlength=n_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    xlabels = [f"C{k}\n{label_lookup.get(k, '')}" for k in range(n_clusters)]
    ax.bar(xlabels, counts, color=[palette[k] for k in range(n_clusters)])
    ax.set_ylabel("Window Count")
    ax.set_title(f"Cluster Sizes [{tag}  K={n_clusters}]")
    for i, c in enumerate(counts):
        ax.text(i, c + 10, str(c), ha="center", fontsize=9)
    _savefig(fig, out_dir / f"kmeans_sizes_{tag}_K{n_clusters}.png")

    return labels, {
        "silhouette": float(final_sil),
        "sil_ci_lo": float(ci_lo),
        "sil_ci_hi": float(ci_hi),
        "db_score": float(davies_bouldin_score(proj, labels)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7.4  GMM
# ═══════════════════════════════════════════════════════════════════════════

def run_gmm(proj, n_clusters, max_k, tag, out_dir, reference_labels=None):
    print(f"\n[GMM — {tag}  K={n_clusters}]")
    bics, ks = [], range(2, max_k + 1)
    for k in ks:
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            random_state=42,
            max_iter=300,
        )
        gm.fit(proj)
        bics.append(gm.bic(proj))
        print(f"  k={k}  BIC={bics[-1]:.0f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), bics, "o-", color=_DIAG_A)
    ax.axvline(n_clusters, ls="--", color=_DIAG_VLINE, lw=1.5, label=f"K={n_clusters}")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC (lower=better)")
    ax.set_title(f"GMM BIC Curve [{tag}]")
    ax.legend()
    _savefig(fig, out_dir / f"gmm_bic_{tag}.png")

    gm_f = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=10,
        random_state=42,
        max_iter=500,
    )
    gm_f.fit(proj)
    resp = gm_f.predict_proba(proj)
    labels = gm_f.predict(proj)

    if reference_labels is not None:
        labels, _ = align_labels_to_reference(reference_labels, labels, n_clusters)
    else:
        labels, _ = canonicalise_labels_by_centroid(proj, labels, n_clusters)

    palette = cluster_colour_map(n_clusters)
    max_resp = resp.max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in range(n_clusters):
        mask = labels == k
        axes[0].scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=4,
            alpha=0.4,
            color=palette[k],
            label=f"C{k}",
            rasterized=True,
        )
    axes[0].set_xlabel("PC1 (largest variance direction in embedding space)")
    axes[0].set_ylabel("PC2 (second-largest variance direction)")
    axes[0].set_title(f"GMM Hard Assignments (K={n_clusters})")
    axes[0].legend(markerscale=4, fontsize=9)

    sc = axes[1].scatter(
        proj[:, 0],
        proj[:, 1],
        s=4,
        c=max_resp,
        cmap="viridis",
        alpha=0.5,
        rasterized=True,
        vmin=0.5,
        vmax=1.0,
    )
    plt.colorbar(sc, ax=axes[1], label="Max Responsibility γ_k")
    axes[1].set_xlabel("PC1 (largest variance direction in embedding space)")
    axes[1].set_ylabel("PC2 (second-largest variance direction)")
    axes[1].set_title("Assignment Certainty")
    fig.suptitle(f"GMM — Latent Space [{tag}]", fontsize=13)
    _savefig(fig, out_dir / f"gmm_scatter_{tag}_K{n_clusters}.png")

    return labels, resp


# ═══════════════════════════════════════════════════════════════════════════
# 7.5  HMM — transitions + dwell time distributions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_dwell_times(
    labels: np.ndarray,
    n_clusters: int,
    stride: int = 16,
    rate_hz: float = 200.0,
) -> Dict[int, np.ndarray]:
    """
    Extract dwell times (in seconds) for each regime from the label sequence.
    stride=16 means consecutive windows overlap by (128-16) samples.
    Each window step = stride/rate_hz seconds of real time.
    """
    step_sec = stride / rate_hz
    dwells = {k: [] for k in range(n_clusters)}
    i, N = 0, len(labels)
    while i < N:
        k = labels[i]
        j = i + 1
        while j < N and labels[j] == k:
            j += 1
        dwells[k].append((j - i) * step_sec)
        i = j
    return {k: np.array(v) for k, v in dwells.items()}


def run_hmm(labels, n_clusters, n_iter, stride, tag, out_dir, beh_labels):
    print(f"\n[HMM — {tag}  K={n_clusters}]")
    palette = cluster_colour_map(n_clusters)

    obs = labels.reshape(-1, 1)
    model = hmm.CategoricalHMM(
        n_components=n_clusters,
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    model.fit(obs)
    A = model.transmat_

    eigvals, eigvecs = np.linalg.eig(A.T)
    stat = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    stat = np.abs(stat) / np.abs(stat).sum()

    print("  Transition matrix:")
    for i in range(n_clusters):
        row = "  ".join(f"{A[i, j]:.3f}" for j in range(n_clusters))
        print(f"    C{i} [{beh_labels.get(i, '?')}]: [{row}]  persist={A[i, i]:.3f}")
    print(f"  Stationary: {np.round(stat, 3)}")

    # ── transition matrix heatmap ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(A, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0])
    tick_labels = [f"C{k}\n{beh_labels.get(k, '')}" for k in range(n_clusters)]
    axes[0].set_xticks(range(n_clusters))
    axes[0].set_xticklabels(tick_labels, fontsize=7)
    axes[0].set_yticks(range(n_clusters))
    axes[0].set_yticklabels(tick_labels, fontsize=7)
    axes[0].set_xlabel("To State")
    axes[0].set_ylabel("From State")
    axes[0].set_title("HMM Transition Matrix P")
    for i in range(n_clusters):
        for j in range(n_clusters):
            axes[0].text(
                j,
                i,
                f"{A[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if A[i, j] > 0.5 else "black",
            )

    xlabels = [f"C{k}\n{beh_labels.get(k, '')}" for k in range(n_clusters)]
    axes[1].bar(xlabels, stat, color=[palette[k] for k in range(n_clusters)])
    axes[1].set_ylabel("Stationary Probability")
    axes[1].set_title("Stationary Distribution π∞")
    for i, s in enumerate(stat):
        axes[1].text(i, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)
    fig.suptitle(f"HMM Regime Transition Structure [{tag}  K={n_clusters}]", fontsize=13)
    _savefig(fig, out_dir / f"hmm_transitions_{tag}_K{n_clusters}.png")

    # ── persistence bar ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    xlabels_p = [f"C{k}\n{beh_labels.get(k, '')}" for k in range(n_clusters)]
    ax.bar(
        xlabels_p,
        [A[k, k] for k in range(n_clusters)],
        color=[palette[k] for k in range(n_clusters)],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Self-Transition Probability P_ii")
    ax.set_title(f"Regime Persistence [{tag}  K={n_clusters}]")
    for k in range(n_clusters):
        ax.text(k, A[k, k] + 0.01, f"{A[k, k]:.3f}", ha="center", fontsize=9)
    _savefig(fig, out_dir / f"hmm_persistence_{tag}_K{n_clusters}.png")

    # ── transition graph ──────────────────────────────────────────────────
    _plot_transition_graph(
        A,
        n_clusters,
        beh_labels,
        tag,
        out_dir / f"hmm_graph_{tag}_K{n_clusters}.png",
    )

    # ── dwell time distributions ──────────────────────────────────────────
    dwells = _compute_dwell_times(labels, n_clusters, stride=stride)
    _plot_dwell_times(
        dwells,
        n_clusters,
        beh_labels,
        tag,
        out_dir / f"hmm_dwell_{tag}_K{n_clusters}.png",
    )

    np.save(out_dir / f"hmm_transmat_{tag}_K{n_clusters}.npy", A)
    np.save(out_dir / f"hmm_stationary_{tag}_K{n_clusters}.npy", stat)

    dwell_stats = {}
    for k, d in dwells.items():
        if len(d) > 0:
            dwell_stats[k] = {
                "mean_sec": float(d.mean()),
                "median_sec": float(np.median(d)),
                "std_sec": float(d.std()),
                "n_episodes": int(len(d)),
            }
            print(
                f"  C{k} dwell: mean={d.mean():.2f}s  "
                f"median={np.median(d):.2f}s  episodes={len(d)}"
            )

    with (out_dir / f"hmm_dwell_stats_{tag}_K{n_clusters}.json").open("w") as f:
        json.dump(dwell_stats, f, indent=2)

    return A, stat, dwell_stats


def _plot_transition_graph(A, n_clusters, beh_labels, tag, path):
    """
    Draw the HMM as a directed graph. Node size ∝ stationary prob.
    Edge thickness and opacity ∝ transition probability.
    Only edges with P > 0.01 are drawn.
    """
    palette = cluster_colour_map(n_clusters)

    eigvals, eigvecs = np.linalg.eig(A.T)
    stat = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    stat = np.abs(stat) / np.abs(stat).sum()

    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) - np.pi / 2
    pos = {k: (np.cos(angles[k]), np.sin(angles[k])) for k in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title(f"HMM Regime Transition Graph [{tag}  K={n_clusters}]", fontsize=13, pad=20)

    for i in range(n_clusters):
        for j in range(n_clusters):
            p = A[i, j]
            if p < 0.01:
                continue
            xi, yi = pos[i]
            xj, yj = pos[j]
            lw = 1 + 8 * p
            alpha = 0.3 + 0.7 * p
            color = palette[i]

            if i == j:
                arc = matplotlib.patches.FancyArrowPatch(
                    posA=(xi - 0.12, yi + 0.12),
                    posB=(xi + 0.12, yi + 0.12),
                    connectionstyle="arc3,rad=-1.2",
                    arrowstyle=f"->,head_width={0.02 + 0.06 * p},head_length=0.05",
                    color=color,
                    lw=lw * 0.5,
                    alpha=alpha,
                )
                ax.add_patch(arc)
                ax.text(
                    xi,
                    yi + 0.42,
                    f"{p:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    alpha=alpha,
                )
            else:
                ax.annotate(
                    "",
                    xy=(xj * 0.72, yj * 0.72),
                    xytext=(xi * 0.72, yi * 0.72),
                    arrowprops=dict(
                        arrowstyle=f"->,head_width={0.02 + 0.05 * p},head_length=0.08",
                        color=color,
                        lw=lw,
                        alpha=alpha,
                        connectionstyle="arc3,rad=0.15",
                    ),
                )
                mx = (xi * 0.72 + xj * 0.72) / 2 * 1.15
                my = (yi * 0.72 + yj * 0.72) / 2 * 1.15
                ax.text(
                    mx,
                    my,
                    f"{p:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    alpha=max(alpha, 0.5),
                )

    for k in range(n_clusters):
        x, y = pos[k]
        radius = 0.18 + 0.22 * stat[k]
        circle = plt.Circle(
            (x * 0.72, y * 0.72),
            radius,
            color=palette[k],
            zorder=5,
            alpha=0.9,
        )
        ax.add_patch(circle)
        label = beh_labels.get(k, f"C{k}")
        short = label[:16] + "…" if len(label) > 16 else label
        ax.text(
            x * 0.72,
            y * 0.72,
            f"C{k}\n{short}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )
        ax.text(
            x * 1.35,
            y * 1.35,
            f"π={stat[k]:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color=palette[k],
        )

    _savefig(fig, path)


def _plot_dwell_times(dwells, n_clusters, beh_labels, tag, path):
    """
    For each regime: histogram of dwell times + fitted exponential approximation.
    """
    palette = cluster_colour_map(n_clusters)
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4), sharey=False)
    if n_clusters == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        d = dwells.get(k, np.array([]))
        if len(d) == 0:
            ax.set_title(f"C{k} — no data")
            continue

        ax.hist(
            d,
            bins=30,
            color=palette[k],
            alpha=0.75,
            edgecolor="white",
            density=True,
        )

        mean_d = float(d.mean())
        xs = np.linspace(0, np.percentile(d, 98), 200)
        lam = 1.0 / mean_d
        ax.plot(xs, lam * np.exp(-lam * xs), "k--", lw=1.5, label=f"Exp(λ={lam:.2f})")

        label = beh_labels.get(k, f"C{k}")
        short = label[:20]
        ax.set_title(f"C{k}: {short}", fontsize=9)
        ax.set_xlabel("Dwell time (s)")
        ax.set_ylabel("Density")
        ax.text(
            0.97,
            0.97,
            f"μ={mean_d:.2f}s\nn={len(d)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.legend(fontsize=7)

    fig.suptitle(f"Regime Dwell Time Distributions [{tag}  K={n_clusters}]", fontsize=12)
    plt.tight_layout()
    _savefig(fig, path)


# ═══════════════════════════════════════════════════════════════════════════
# 7.6  Behavioural interpretation
# ═══════════════════════════════════════════════════════════════════════════

def run_behavioural(
    emb,
    labels,
    proj,
    n_clusters,
    dataset_path,
    beh_labels,
    tag,
    out_dir,
):
    print(f"\n[Behaviour — {tag}  K={n_clusters}]")
    palette = cluster_colour_map(n_clusters)

    X_all, mu, sigma, _ = _load_states(dataset_path)
    s_norm = X_all[:, -1, :]
    s_raw = s_norm * sigma[None, :] + mu[None, :]

    dp_mag = np.linalg.norm(s_raw[:, 0:3], axis=1)
    v_mag = np.linalg.norm(s_raw[:, 3:6], axis=1)
    w_mag = np.linalg.norm(s_raw[:, 10:13], axis=1)
    ac_mag = np.linalg.norm(s_raw[:, 13:16], axis=1)

    # ── kinematic box plots ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    proxies = [
        (dp_mag, "||Δp|| (m)", "m"),
        (v_mag, "||v|| (m/s)", "m/s"),
        (w_mag, "||ω|| (rad/s)", "rad/s"),
        (ac_mag, "||a_comp|| (m/s²)", "m/s²"),
    ]
    for ax, (mag, title, unit) in zip(axes, proxies):
        data = [mag[labels == k] for k in range(n_clusters)]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="white", lw=2),
        )
        for patch, k in zip(bp["boxes"], range(n_clusters)):
            patch.set_facecolor(palette[k])
            patch.set_alpha(0.85)
        xlabels = [f"C{k}\n{beh_labels.get(k, '')}" for k in range(n_clusters)]
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel(unit)
        ax.set_title(title)
    fig.suptitle(
        f"Cluster Kinematics — Behavioural Proxies [{tag}  K={n_clusters}]",
        fontsize=13,
    )
    _savefig(fig, out_dir / f"behaviour_magnitudes_{tag}_K{n_clusters}.png")

    # ── feature mean heatmap ──────────────────────────────────────────────
    fn = FEATURE_NAMES if s_norm.shape[1] == 16 else [f"f{i}" for i in range(s_norm.shape[1])]
    means = np.array([s_norm[labels == k].mean(axis=0) for k in range(n_clusters)])
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(means, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="Normalised mean")
    ax.set_xticks(range(len(fn)))
    ax.set_xticklabels(fn, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"C{k}: {beh_labels.get(k, '')}" for k in range(n_clusters)], fontsize=8)
    ax.set_title(f"Cluster Mean State Vectors [{tag}  K={n_clusters}]")
    _savefig(fig, out_dir / f"behaviour_heatmap_{tag}_K{n_clusters}.png")

    # ── labelled scatter ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=4,
            alpha=0.35,
            color=palette[k],
            label=f"C{k}: {beh_labels.get(k, str(k))}",
            rasterized=True,
        )
    ax.set_xlabel("PC1 (largest variance direction in embedding space)")
    ax.set_ylabel("PC2 (second-largest variance direction)")
    ax.set_title(f"Latent Space — Behavioural Regimes [{tag}  K={n_clusters}]")
    ax.legend(markerscale=4, fontsize=8, loc="upper right")
    _savefig(fig, out_dir / f"behaviour_scatter_{tag}_K{n_clusters}.png")

    # ── summary table ─────────────────────────────────────────────────────
    rows = []
    print(f"\n  {'C':>3}  {'label':>30}  {'v̄':>8}  {'ω̄':>8}  {'ac̄':>8}  {'n':>6}")
    for k in range(n_clusters):
        mask = labels == k
        row = dict(
            cluster=k,
            label=beh_labels.get(k, f"C{k}"),
            v_mean=float(v_mag[mask].mean()),
            v_std=float(v_mag[mask].std()),
            w_mean=float(w_mag[mask].mean()),
            w_std=float(w_mag[mask].std()),
            ac_mean=float(ac_mag[mask].mean()),
            ac_std=float(ac_mag[mask].std()),
            dp_mean=float(dp_mag[mask].mean()),
            n=int(mask.sum()),
        )
        rows.append(row)
        print(
            f"  C{k}  {row['label']:>30}  {row['v_mean']:>8.3f}"
            f"  {row['w_mean']:>8.3f}  {row['ac_mean']:>8.2f}  {row['n']:>6}"
        )
    with (out_dir / f"cluster_kinematics_{tag}_K{n_clusters}.json").open("w") as f:
        json.dump(rows, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 7.7  Sequence-level behavioural profiles
# ═══════════════════════════════════════════════════════════════════════════

def run_sequence_profiles(labels, seq_origin, n_clusters, beh_labels, tag, out_dir):
    if seq_origin is None:
        print("  [seq profiles] seq_origin not available — skipping")
        return

    print(f"\n[Sequence profiles — {tag}  K={n_clusters}]")
    palette = cluster_colour_map(n_clusters)

    sequences = sorted(np.unique(seq_origin))
    if len(sequences) == 0:
        print("  No sequence labels found.")
        return

    # ── per-sequence regime occupancy ─────────────────────────────────────
    occupancy = np.zeros((len(sequences), n_clusters))
    for si, seq in enumerate(sequences):
        mask = seq_origin == seq
        if not mask.any():
            continue
        counts = np.bincount(labels[mask], minlength=n_clusters)
        occupancy[si] = counts / counts.sum()

    # ── stacked bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(sequences) * 1.2), 5))
    bottom = np.zeros(len(sequences))
    x = np.arange(len(sequences))
    for k in range(n_clusters):
        ax.bar(
            x,
            occupancy[:, k],
            bottom=bottom,
            color=palette[k],
            label=f"C{k}: {beh_labels.get(k, '')}",
            alpha=0.85,
        )
        bottom += occupancy[:, k]
    ax.set_ylabel("Regime Occupancy Fraction")
    ax.set_title(f"Per-Sequence Behavioural Regime Occupancy [{tag}  K={n_clusters}]")
    ax.set_xticks(x)
    ax.set_xticklabels(sequences, rotation=45, ha="right", fontsize=8)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _savefig(fig, out_dir / f"seq_occupancy_{tag}_K{n_clusters}.png")

    # ── difficulty comparison ─────────────────────────────────────────────
    diff_groups = {"easy": [], "medium": [], "difficult": []}
    env_groups = {"MH": [], "Vicon": []}
    for si, seq in enumerate(sequences):
        diff_groups[_difficulty(seq)].append(occupancy[si])
        env_groups[_environment(seq)].append(occupancy[si])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, groups, title in [
        (axes[0], diff_groups, "Difficulty Level"),
        (axes[1], env_groups, "Environment"),
    ]:
        valid = {g: np.array(v) for g, v in groups.items() if len(v) > 0}
        gnames = sorted(valid.keys())
        x = np.arange(len(gnames))
        w = 0.8 / n_clusters
        for k in range(n_clusters):
            means = [valid[g][:, k].mean() for g in gnames]
            stds = [valid[g][:, k].std() for g in gnames]
            ax.bar(
                x + k * w - 0.4 + w / 2,
                means,
                w,
                color=palette[k],
                alpha=0.85,
                label=f"C{k}: {beh_labels.get(k, '')}",
                yerr=stds,
                capsize=3,
                error_kw=dict(lw=1),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(gnames, fontsize=9)
        ax.set_ylabel("Mean Regime Occupancy")
        ax.set_title(f"Regime Occupancy by {title}")
        ax.legend(fontsize=7)
    fig.suptitle(f"Behavioural Profiles by Sequence Type [{tag}  K={n_clusters}]", fontsize=12)
    _savefig(fig, out_dir / f"seq_difficulty_{tag}_K{n_clusters}.png")

    # ── numerical summary ─────────────────────────────────────────────────
    summary = {}
    for si, seq in enumerate(sequences):
        summary[seq] = {f"C{k}": float(occupancy[si, k]) for k in range(n_clusters)}
        summary[seq]["difficulty"] = _difficulty(seq)
        summary[seq]["environment"] = _environment(seq)
        dominant = int(np.argmax(occupancy[si]))
        summary[seq]["dominant_regime"] = f"C{dominant}: {beh_labels.get(dominant, '')}"
        print(
            f"  {seq:<20} {_difficulty(seq):<10} "
            f"dominant={summary[seq]['dominant_regime'][:30]}  "
            f"occ={np.round(occupancy[si], 2)}"
        )

    with (out_dir / f"seq_profiles_{tag}_K{n_clusters}.json").open("w") as f:
        json.dump(summary, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Embedding trajectory through one sequence
# ═══════════════════════════════════════════════════════════════════════════

def run_trajectory_viz(proj, labels, seq_origin, n_clusters, beh_labels, tag, out_dir):
    if seq_origin is None:
        print("  [trajectory] seq_origin not available — skipping")
        return

    palette = cluster_colour_map(n_clusters)

    seqs, counts = np.unique(seq_origin, return_counts=True)
    candidates = [(s, c) for s, c in zip(seqs, counts) if c > 50]
    if not candidates:
        print("  [trajectory] no sequences with >50 windows — skipping")
        return

    sorted_cands = sorted(
        candidates,
        key=lambda x: (
            {"easy": 0, "medium": 1, "difficult": 2}[_difficulty(x[0])],
            -x[1],
        ),
    )
    easy_seqs = [s for s, _ in sorted_cands if _difficulty(s) == "easy"]
    hard_seqs = [s for s, _ in sorted_cands if _difficulty(s) in ("medium", "difficult")]
    plot_seqs = []
    if easy_seqs:
        plot_seqs.append(easy_seqs[0])
    if hard_seqs:
        plot_seqs.append(hard_seqs[0])
    if not plot_seqs:
        plot_seqs = [sorted_cands[0][0]]

    print(f"\n[Trajectory — {tag}  K={n_clusters}]")
    for seq in plot_seqs:
        mask = seq_origin == seq
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        p_seq = proj[idx]
        l_seq = labels[idx]
        t_seq = np.arange(len(idx))

        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])

        ax1 = fig.add_subplot(gs[0])
        sc = ax1.scatter(
            p_seq[:, 0],
            p_seq[:, 1],
            c=t_seq,
            cmap="plasma",
            s=8,
            alpha=0.7,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax1, label="Window index (time →)")
        for k in range(n_clusters):
            m = l_seq == k
            if m.any():
                ax1.scatter(
                    p_seq[m, 0],
                    p_seq[m, 1],
                    s=2,
                    alpha=0.2,
                    color=palette[k],
                    rasterized=True,
                )
        ax1.set_xlabel("PC1 (largest variance direction in embedding space)")
        ax1.set_ylabel("PC2 (second-largest variance direction)")
        ax1.set_title(f"PC Trajectory — {seq}")

        ax2 = fig.add_subplot(gs[1])
        for k in range(n_clusters):
            m = l_seq == k
            ax2.scatter(
                t_seq[m],
                np.full(m.sum(), k),
                color=palette[k],
                s=6,
                alpha=0.8,
                label=f"C{k}: {beh_labels.get(k, '')}",
            )
        ax2.set_xlabel("Window index")
        ax2.set_ylabel("Cluster")
        ax2.set_yticks(range(n_clusters))
        ax2.set_yticklabels([f"C{k}" for k in range(n_clusters)])
        ax2.set_title("Regime Sequence Over Time")
        ax2.legend(fontsize=6, loc="upper right")

        fig.suptitle(
            f"Embedding Trajectory: {seq} "
            f"[{_difficulty(seq)}, {_environment(seq)}]",
            fontsize=12,
        )
        safe = seq.replace("/", "_")
        _savefig(fig, out_dir / f"trajectory_{tag}_K{n_clusters}_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Appendix: random baseline ablation
# ═══════════════════════════════════════════════════════════════════════════

def run_random_baseline(proj, labels, n_clusters, tag, out_dir):
    print(f"\n[Random baseline — {tag}  K={n_clusters}]")
    N, D = proj.shape
    n_trials = 50
    rng = np.random.default_rng(0)

    real_sil = silhouette_score(proj, labels, sample_size=min(5000, N))
    rand_sils = []
    for _ in range(n_trials):
        rand_emb = rng.standard_normal((N, D))
        rand_labs = KMeans(
            n_clusters=n_clusters,
            n_init=5,
            random_state=int(rng.integers(0, 10000)),
        ).fit_predict(rand_emb)
        rand_sils.append(
            silhouette_score(rand_emb, rand_labs, sample_size=min(5000, N))
        )
    rand_sils = np.array(rand_sils)
    z_score = (real_sil - rand_sils.mean()) / rand_sils.std()
    p_approx = float((rand_sils >= real_sil).mean())

    print(f"  Real silhouette:   {real_sil:.4f}")
    print(f"  Random mean±std:   {rand_sils.mean():.4f} ± {rand_sils.std():.4f}")
    print(f"  Z-score:           {z_score:.2f}")
    print(
        f"  Empirical p-value: {p_approx:.4f}  "
        f"({'significant' if p_approx < 0.05 else 'not significant'})"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        rand_sils,
        bins=20,
        color=_DIAG_A,
        alpha=0.75,
        edgecolor="white",
        label=f"Random baseline (n={n_trials})",
    )
    ax.axvline(
        real_sil,
        color=_DIAG_VLINE,
        lw=2.5,
        label=f"Transformer embeddings\nsil={real_sil:.4f}  z={z_score:.1f}",
    )
    ax.set_xlabel("Silhouette Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Random Baseline Ablation [{tag}  K={n_clusters}]")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / f"ablation_random_{tag}_K{n_clusters}.png")

    result = {
        "real_silhouette": float(real_sil),
        "random_mean": float(rand_sils.mean()),
        "random_std": float(rand_sils.std()),
        "z_score": float(z_score),
        "p_empirical": p_approx,
    }
    with (out_dir / f"ablation_stats_{tag}_K{n_clusters}.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  MT4599 — Full Latent-State Analysis  (v4)")
    print("=" * 65)

    emb_last, emb_mean, split_labels, seq_origin, meta = _load_embeddings(Path(args.embeddings))
    dataset_path = Path(args.dataset)

    print(f"\nEmbeddings: N={emb_last.shape[0]}  d={emb_last.shape[1]}")
    seqs_present = np.unique(seq_origin) if seq_origin is not None else []
    print(f"Sequences: {list(seqs_present)}")

    K = args.n_clusters
    dpc = args.pca_components

    all_metrics = {}

    # ── PRIMARY: emb_mean, K=4 ────────────────────────────────────────────
    tag = "mean"
    proj_mean, _ = run_pca(emb_mean, dpc, split_labels, tag, out_dir)
    np.save(out_dir / f"_proj_{tag}.npy", proj_mean)

    km_labels_mean4, metrics4 = run_kmeans(
        proj_mean,
        K,
        args.max_k,
        args.n_bootstrap,
        tag,
        out_dir,
        dataset_path,
    )
    np.save(out_dir / f"kmeans_labels_{tag}_K{K}.npy", km_labels_mean4)
    all_metrics[f"{tag}_K{K}"] = metrics4

    run_gmm(
        proj_mean,
        K,
        args.max_k,
        tag,
        out_dir,
        reference_labels=km_labels_mean4,
    )

    A_mean4, stat_mean4, dwell_mean4 = run_hmm(
        km_labels_mean4,
        K,
        args.hmm_n_iter,
        16,
        tag,
        out_dir,
        BEHAVIOUR_LABELS_K4,
    )

    run_behavioural(
        emb_mean,
        km_labels_mean4,
        proj_mean,
        K,
        dataset_path,
        BEHAVIOUR_LABELS_K4,
        tag,
        out_dir,
    )

    run_sequence_profiles(
        km_labels_mean4,
        seq_origin,
        K,
        BEHAVIOUR_LABELS_K4,
        tag,
        out_dir,
    )

    run_trajectory_viz(
        proj_mean,
        km_labels_mean4,
        seq_origin,
        K,
        BEHAVIOUR_LABELS_K4,
        tag,
        out_dir,
    )

    ablation_mean4 = run_random_baseline(proj_mean, km_labels_mean4, K, tag, out_dir)
    all_metrics[f"{tag}_K{K}"]["ablation"] = ablation_mean4

    # ── SECONDARY: emb_last, K=3 (robustness check) ───────────────────────
    tag3 = "last"
    proj_last, _ = run_pca(emb_last, dpc, split_labels, tag3, out_dir)
    np.save(out_dir / f"_proj_{tag3}.npy", proj_last)

    km_labels_last3, metrics3 = run_kmeans(
        proj_last,
        3,
        args.max_k,
        args.n_bootstrap,
        tag3,
        out_dir,
        dataset_path,
    )
    np.save(out_dir / f"kmeans_labels_{tag3}_K3.npy", km_labels_last3)
    all_metrics[f"{tag3}_K3"] = metrics3

    run_hmm(
        km_labels_last3,
        3,
        args.hmm_n_iter,
        16,
        tag3,
        out_dir,
        BEHAVIOUR_LABELS_K3,
    )

    run_behavioural(
        emb_last,
        km_labels_last3,
        proj_last,
        3,
        dataset_path,
        BEHAVIOUR_LABELS_K3,
        tag3,
        out_dir,
    )

    run_sequence_profiles(
        km_labels_last3,
        seq_origin,
        3,
        BEHAVIOUR_LABELS_K3,
        tag3,
        out_dir,
    )

    run_trajectory_viz(
        proj_last,
        km_labels_last3,
        seq_origin,
        3,
        BEHAVIOUR_LABELS_K3,
        tag3,
        out_dir,
    )

    ablation_last3 = run_random_baseline(proj_last, km_labels_last3, 3, tag3, out_dir)
    all_metrics[f"{tag3}_K3"]["ablation"] = ablation_last3

    # ── Save all metrics ───────────────────────────────────────────────────
    cfg = vars(args)
    cfg["all_metrics"] = all_metrics
    with (out_dir / "analysis_summary.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Done. Outputs in: {out_dir}")
    print("=" * 65)

    print("\n── Metric Summary ──")
    for name, m in all_metrics.items():
        sil = m.get("silhouette", float("nan"))
        lo = m.get("sil_ci_lo", float("nan"))
        hi = m.get("sil_ci_hi", float("nan"))
        db = m.get("db_score", float("nan"))
        z = m.get("ablation", {}).get("z_score", float("nan"))
        print(
            f"  {name:<12}  sil={sil:.4f} [{lo:.4f},{hi:.4f}]  "
            f"DB={db:.4f}  ablation_z={z:.1f}"
        )


if __name__ == "__main__":
    main()