"""
analyse_embeddings.py  —  v2
============================
Statistical latent-state analysis of transformer encoder embeddings.
Follows thesis Sections 5.4–5.7.

Changes vs v1:
- Uses mean-pooled sequence embedding (emb_seq.mean over time axis) as well as
  emb_last; both are analysed and saved separately so results can be compared.
- K=3 analysis run alongside K=4 (silhouette-preferred vs theory-preferred).
- Gravity-compensated acceleration in kinematic profiles (ac_x/y/z instead of a_x/y/z).
- Position increments (dp) instead of absolute position in feature heatmap.
- All figures saved at 200 dpi.
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
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

warnings.filterwarnings("ignore")

CLUSTER_COLOURS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
                   "#F4A261", "#264653", "#A8DADC", "#6D6875"]

# Feature names for the improved state vector (gravity-compensated, pos increments)
FEATURE_NAMES = ["dp_x","dp_y","dp_z",
                 "v_x","v_y","v_z",
                 "q_x","q_y","q_z","q_w",
                 "w_x","w_y","w_z",
                 "ac_x","ac_y","ac_z"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings",     required=True)
    p.add_argument("--dataset",        required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--n-clusters",     type=int, default=4)
    p.add_argument("--pca-components", type=int, default=32)
    p.add_argument("--hmm-n-iter",     type=int, default=200)
    p.add_argument("--max-k",          type=int, default=8)
    return p.parse_args()


def _savefig(fig, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def _load_embeddings(path: Path):
    npz  = np.load(path, allow_pickle=False)
    meta = json.loads(str(npz["meta_json"]))
    labs = npz["split_labels"].astype(str)

    # emb_last: (N, d_model)
    emb_last = npz["emb_last"].astype(np.float32)

    # emb_seq mean-pool: (N, d_model)  — average over the time dimension
    if "emb_seq" in npz:
        emb_mean = npz["emb_seq"].astype(np.float32).mean(axis=1)
    else:
        emb_mean = emb_last   # fallback

    return emb_last, emb_mean, labs, meta


def _load_states(dataset_path: Path):
    """Load all normalised windows and mu/sigma for inverse transform."""
    npz     = np.load(dataset_path, allow_pickle=False)
    X_all   = np.concatenate([npz["X_train"], npz["X_val"], npz["X_test"]], axis=0)
    mu      = npz["mu"].astype(np.float64)
    sigma   = npz["sigma"].astype(np.float64)
    return X_all, mu, sigma


# ─── PCA ─────────────────────────────────────────────────────────────────────

def run_pca(emb, n_components, split_labels, tag, out_dir):
    print(f"\n[PCA — {tag}]")
    scaler   = StandardScaler()
    emb_s    = scaler.fit_transform(emb)
    pca_full = PCA(n_components=min(emb.shape[1], 64)).fit(emb_s)
    pca      = PCA(n_components=n_components).fit(emb_s)
    proj     = pca.transform(emb_s)
    var      = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA({n_components}) retains {var*100:.1f}% variance")

    explained  = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, min(33, len(explained)+1)), explained[:32]*100, color="#457B9D", alpha=0.8)
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot")
    axes[1].plot(range(1, len(cumulative)+1), cumulative*100, "o-", color="#E63946", ms=4)
    axes[1].axhline(90, ls="--", color="gray", lw=1, label="90%")
    axes[1].axhline(95, ls=":",  color="gray", lw=1, label="95%")
    axes[1].axvline(n_components, ls="--", color="#2A9D8F", lw=1.5,
                    label=f"d={n_components} ({var*100:.0f}%)")
    axes[1].set_xlabel("Components"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance"); axes[1].legend(fontsize=8)
    axes[1].set_xlim(0.5, min(64, len(cumulative))+0.5); axes[1].set_ylim(0, 102)
    fig.suptitle(f"PCA of Encoder Embeddings [{tag}]", fontsize=13)
    _savefig(fig, out_dir / f"pca_scree_{tag}.png")

    split_colour = {"train": "#457B9D", "val": "#2A9D8F", "test": "#E63946"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in ["train", "val", "test"]:
        mask = split_labels == sp
        if mask.sum() == 0: continue
        ax.scatter(proj[mask,0], proj[mask,1], s=4, alpha=0.4,
                   color=split_colour[sp], label=sp, rasterized=True)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"Embedding Space — PC1 vs PC2 [{tag}]")
    ax.legend(markerscale=4, fontsize=9)
    _savefig(fig, out_dir / f"pca_scatter_split_{tag}.png")

    return proj, pca


# ─── K-means ─────────────────────────────────────────────────────────────────

def run_kmeans(proj, n_clusters, max_k, tag, out_dir):
    print(f"\n[K-means — {tag}  K={n_clusters}]")
    inertias, sils, ks = [], [], range(2, max_k+1)
    for k in ks:
        km  = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbs = km.fit_predict(proj)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(proj, lbs, sample_size=min(5000, len(lbs))))
        print(f"  k={k}  inertia={km.inertia_:.1f}  sil={sils[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(ks), inertias, "o-", color="#457B9D")
    axes[0].axvline(n_clusters, ls="--", color="#E63946", lw=1.5, label=f"K={n_clusters}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Curve"); axes[0].legend()
    axes[1].plot(list(ks), sils, "o-", color="#2A9D8F")
    axes[1].axvline(n_clusters, ls="--", color="#E63946", lw=1.5, label=f"K={n_clusters}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores"); axes[1].legend()
    fig.suptitle(f"K-means Cluster Selection [{tag}]", fontsize=13)
    _savefig(fig, out_dir / f"kmeans_selection_{tag}.png")

    km_final = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels   = km_final.fit_predict(proj)
    final_sil = silhouette_score(proj, labels, sample_size=min(5000, len(labels)))
    print(f"  Final K={n_clusters} silhouette={final_sil:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(proj[mask,0], proj[mask,1], s=4, alpha=0.4,
                   color=CLUSTER_COLOURS[k], label=f"C{k}", rasterized=True)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"K-means (K={n_clusters}) — PC1 vs PC2 [{tag}]")
    ax.legend(markerscale=4, fontsize=9)
    _savefig(fig, out_dir / f"kmeans_scatter_{tag}_K{n_clusters}.png")

    counts = np.bincount(labels, minlength=n_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"C{k}" for k in range(n_clusters)], counts, color=CLUSTER_COLOURS[:n_clusters])
    ax.set_ylabel("Window Count"); ax.set_title(f"Cluster Sizes [{tag}  K={n_clusters}]")
    for i, c in enumerate(counts):
        ax.text(i, c+10, str(c), ha="center", fontsize=9)
    _savefig(fig, out_dir / f"kmeans_sizes_{tag}_K{n_clusters}.png")

    return labels


# ─── GMM ─────────────────────────────────────────────────────────────────────

def run_gmm(proj, n_clusters, max_k, tag, out_dir):
    print(f"\n[GMM — {tag}  K={n_clusters}]")
    bics, ks = [], range(2, max_k+1)
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, random_state=42, max_iter=300)
        gm.fit(proj)
        bics.append(gm.bic(proj))
        print(f"  k={k}  BIC={bics[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), bics, "o-", color="#457B9D")
    ax.axvline(n_clusters, ls="--", color="#E63946", lw=1.5, label=f"K={n_clusters}")
    ax.set_xlabel("K"); ax.set_ylabel("BIC")
    ax.set_title(f"GMM BIC Curve [{tag}]"); ax.legend()
    _savefig(fig, out_dir / f"gmm_bic_{tag}.png")

    gm_final = GaussianMixture(n_components=n_clusters, covariance_type="full",
                               n_init=10, random_state=42, max_iter=500)
    gm_final.fit(proj)
    resp   = gm_final.predict_proba(proj)
    labels = gm_final.predict(proj)

    max_resp = resp.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in range(n_clusters):
        mask = labels == k
        axes[0].scatter(proj[mask,0], proj[mask,1], s=4, alpha=0.4,
                        color=CLUSTER_COLOURS[k], label=f"C{k}", rasterized=True)
    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
    axes[0].set_title(f"GMM Hard Assignments (K={n_clusters})")
    axes[0].legend(markerscale=4, fontsize=9)
    sc = axes[1].scatter(proj[:,0], proj[:,1], s=4, c=max_resp, cmap="viridis",
                         alpha=0.5, rasterized=True, vmin=0.5, vmax=1.0)
    plt.colorbar(sc, ax=axes[1], label="Max Responsibility γ_k")
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
    axes[1].set_title("Assignment Certainty")
    fig.suptitle(f"GMM — Latent Space [{tag}]", fontsize=13)
    _savefig(fig, out_dir / f"gmm_scatter_{tag}_K{n_clusters}.png")

    entropy = -np.sum(resp * np.log(resp + 1e-12), axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"C{k}" for k in range(n_clusters)],
           [entropy[labels==k].mean() for k in range(n_clusters)],
           color=CLUSTER_COLOURS[:n_clusters])
    ax.set_ylabel("Mean Assignment Entropy")
    ax.set_title(f"GMM Uncertainty per Cluster [{tag}  K={n_clusters}]")
    _savefig(fig, out_dir / f"gmm_uncertainty_{tag}_K{n_clusters}.png")

    return labels, resp


# ─── HMM ─────────────────────────────────────────────────────────────────────

def run_hmm(labels, n_clusters, n_iter, tag, out_dir):
    print(f"\n[HMM — {tag}  K={n_clusters}]")
    obs   = labels.reshape(-1, 1)
    model = hmm.CategoricalHMM(n_components=n_clusters, n_iter=n_iter,
                               random_state=42, verbose=False)
    model.fit(obs)
    A = model.transmat_

    eigvals, eigvecs = np.linalg.eig(A.T)
    stat = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    stat = np.abs(stat) / np.abs(stat).sum()

    print("  Transition matrix:")
    for i in range(n_clusters):
        row = "  ".join(f"{A[i,j]:.3f}" for j in range(n_clusters))
        print(f"    C{i}: [{row}]  persist={A[i,i]:.3f}")
    print(f"  Stationary: {np.round(stat,3)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(A, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks(range(n_clusters)); axes[0].set_yticks(range(n_clusters))
    axes[0].set_xticklabels([f"C{k}" for k in range(n_clusters)])
    axes[0].set_yticklabels([f"C{k}" for k in range(n_clusters)])
    axes[0].set_xlabel("To State"); axes[0].set_ylabel("From State")
    axes[0].set_title("HMM Transition Matrix P")
    for i in range(n_clusters):
        for j in range(n_clusters):
            axes[0].text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if A[i,j]>0.5 else "black")
    axes[1].bar([f"C{k}" for k in range(n_clusters)], stat,
                color=CLUSTER_COLOURS[:n_clusters])
    axes[1].set_ylabel("Stationary Probability")
    axes[1].set_title("Stationary Distribution π∞")
    for i, s in enumerate(stat):
        axes[1].text(i, s+0.005, f"{s:.3f}", ha="center", fontsize=9)
    fig.suptitle(f"HMM — Regime Transition Structure [{tag}  K={n_clusters}]", fontsize=13)
    _savefig(fig, out_dir / f"hmm_transitions_{tag}_K{n_clusters}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"C{k}" for k in range(n_clusters)],
           [A[k,k] for k in range(n_clusters)], color=CLUSTER_COLOURS[:n_clusters])
    ax.set_ylim(0, 1); ax.set_ylabel("Self-Transition P_ii")
    ax.set_title(f"Regime Persistence [{tag}  K={n_clusters}]")
    for k in range(n_clusters):
        ax.text(k, A[k,k]+0.01, f"{A[k,k]:.3f}", ha="center", fontsize=9)
    _savefig(fig, out_dir / f"hmm_persistence_{tag}_K{n_clusters}.png")

    np.save(out_dir / f"hmm_transmat_{tag}_K{n_clusters}.npy", A)
    np.save(out_dir / f"hmm_stationary_{tag}_K{n_clusters}.npy", stat)


# ─── Behavioural interpretation ───────────────────────────────────────────────

def run_behavioural(emb, labels, proj, n_clusters, dataset_path, tag, out_dir):
    print(f"\n[Behaviour — {tag}  K={n_clusters}]")
    X_all, mu, sigma = _load_states(dataset_path)

    # Use last timestep of each window as representative state
    s_norm = X_all[:, -1, :]                          # (N, 16) normalised
    s_raw  = s_norm * sigma[None,:] + mu[None,:]       # (N, 16) original units

    # With the new state vector: indices for dp(0:3), v(3:6), q(6:10), w(10:13), ac(13:16)
    dp_mag = np.linalg.norm(s_raw[:, 0:3],  axis=1)   # ||Δp||
    v_mag  = np.linalg.norm(s_raw[:, 3:6],  axis=1)   # ||v||
    w_mag  = np.linalg.norm(s_raw[:, 10:13], axis=1)  # ||ω||
    ac_mag = np.linalg.norm(s_raw[:, 13:16], axis=1)  # ||a_comp|| (gravity-free)

    # ── kinematic box plots ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, mag, title, unit in zip(
        axes,
        [dp_mag,    v_mag,      w_mag,           ac_mag],
        ["||Δp|| (m)", "||v|| (m/s)", "||ω|| (rad/s)", "||a_comp|| (m/s²)"],
        ["m",       "m/s",      "rad/s",         "m/s²"],
    ):
        data = [mag[labels==k] for k in range(n_clusters)]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="white", lw=2))
        for patch, col in zip(bp["boxes"], CLUSTER_COLOURS[:n_clusters]):
            patch.set_facecolor(col); patch.set_alpha(0.8)
        ax.set_xticklabels([f"C{k}" for k in range(n_clusters)])
        ax.set_ylabel(unit); ax.set_title(title)
    fig.suptitle(f"Cluster Kinematics — Behavioural Proxies [{tag}  K={n_clusters}]", fontsize=13)
    _savefig(fig, out_dir / f"behaviour_magnitudes_{tag}_K{n_clusters}.png")

    # ── feature heatmap ───────────────────────────────────────────────────
    cluster_means = np.array([s_norm[labels==k].mean(axis=0) for k in range(n_clusters)])
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(cluster_means, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="Normalised mean")
    fn = FEATURE_NAMES if s_norm.shape[1] == 16 else [f"f{i}" for i in range(s_norm.shape[1])]
    ax.set_xticks(range(len(fn))); ax.set_xticklabels(fn, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {k}" for k in range(n_clusters)])
    ax.set_title(f"Cluster Mean State Vectors [{tag}  K={n_clusters}]")
    _savefig(fig, out_dir / f"behaviour_heatmap_{tag}_K{n_clusters}.png")

    # ── summary + tentative labels ────────────────────────────────────────
    rows = []
    print(f"  {'C':>3}  {'||Δp||':>10}  {'||v||':>10}  {'||ω||':>10}  {'||ac||':>10}  {'N':>6}")
    for k in range(n_clusters):
        mask = labels == k
        row  = dict(cluster=k,
                    dp_mean=float(dp_mag[mask].mean()), dp_std=float(dp_mag[mask].std()),
                    v_mean =float(v_mag[mask].mean()),  v_std =float(v_mag[mask].std()),
                    w_mean =float(w_mag[mask].mean()),  w_std =float(w_mag[mask].std()),
                    ac_mean=float(ac_mag[mask].mean()), ac_std=float(ac_mag[mask].std()),
                    n=int(mask.sum()))
        rows.append(row)
        print(f"  C{k}  {row['dp_mean']:>8.4f}  {row['v_mean']:>8.4f}  "
              f"{row['w_mean']:>8.4f}  {row['ac_mean']:>8.4f}  {row['n']:>6}")

    with (out_dir / f"cluster_kinematics_{tag}_K{n_clusters}.json").open("w") as f:
        json.dump(rows, f, indent=2)

    # Rank by gravity-compensated acceleration magnitude
    ac_means = [ac_mag[labels==k].mean() for k in range(n_clusters)]
    order    = np.argsort(ac_means)
    if n_clusters == 4:
        tentative = {int(order[0]): "Hover / Steady State",
                     int(order[1]): "Low-speed Inspection",
                     int(order[2]): "Moderate Motion",
                     int(order[3]): "Aggressive Manoeuvre"}
    else:
        tentative = {int(order[i]): f"Regime {i+1} (ac̄={ac_means[order[i]]:.2f} m/s²)"
                     for i in range(n_clusters)}

    print("\n  Tentative behavioural labels (ac-ranked):")
    for k, lbl in sorted(tentative.items()):
        print(f"    C{k}: {lbl}")

    with (out_dir / f"tentative_labels_{tag}_K{n_clusters}.json").open("w") as f:
        json.dump(tentative, f, indent=2)

    # ── labelled scatter ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(proj[mask,0], proj[mask,1], s=4, alpha=0.35,
                   color=CLUSTER_COLOURS[k],
                   label=f"C{k}: {tentative.get(k, str(k))}", rasterized=True)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"Latent Space — Behavioural Regimes [{tag}  K={n_clusters}]")
    ax.legend(markerscale=4, fontsize=8, loc="upper right")
    _savefig(fig, out_dir / f"behaviour_scatter_{tag}_K{n_clusters}.png")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Latent-State Analysis  (v2 — gravity-comp / pos-increments)")
    print("=" * 65)

    emb_last, emb_mean, split_labels, meta = _load_embeddings(Path(args.embeddings))
    dataset_path = Path(args.dataset)

    print(f"\nEmbeddings loaded: N={emb_last.shape[0]}  d={emb_last.shape[1]}")
    print(f"Splits — train={np.sum(split_labels=='train')}  "
          f"val={np.sum(split_labels=='val')}  "
          f"test={np.sum(split_labels=='test')}")

    K    = args.n_clusters
    dpc  = args.pca_components

    # Run analysis for both embedding types and both K=3, K=4
    for emb, tag in [(emb_last, "last"), (emb_mean, "mean")]:
        proj, _ = run_pca(emb, dpc, split_labels, tag, out_dir)
        np.save(out_dir / f"_proj_{tag}.npy", proj)

        for k in sorted(set([3, K])):
            km_labels = run_kmeans(proj, k, args.max_k, tag, out_dir)
            np.save(out_dir / f"kmeans_labels_{tag}_K{k}.npy", km_labels)

            gmm_labels, _ = run_gmm(proj, k, args.max_k, tag, out_dir)
            np.save(out_dir / f"gmm_labels_{tag}_K{k}.npy", gmm_labels)

            run_hmm(km_labels, k, args.hmm_n_iter, tag, out_dir)
            run_behavioural(emb, km_labels, proj, k, dataset_path, tag, out_dir)

    cfg = vars(args)
    cfg["n_embeddings"] = int(emb_last.shape[0])
    cfg["d_model"]      = int(emb_last.shape[1])
    with (out_dir / "analysis_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Done. All outputs in: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
