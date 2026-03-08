mt4599: EuRoC MAV preprocessing for thesis
==========================================

This repo contains preprocessing code for a thesis project that trains transformer models on EuRoC MAV time series and then analyses the learned embeddings.

The first milestone is a **single-sequence** preprocessing pipeline that:

- Loads IMU and pose / ground-truth data from a EuRoC sequence folder (e.g. `V1_01_easy`).
- Resamples both streams to a **uniform 200 Hz grid** (IMU body frame preserved).
- Constructs state vectors
  - \(s_t = [p_t, v_t, q_t, \omega_t, a_t]\)
  - position \(p_t \in \mathbb{R}^3\)
  - velocity \(v_t \in \mathbb{R}^3\) (from pose finite differences)
  - orientation quaternion \(q_t \in \mathbb{R}^4\)
  - angular velocity \(\omega_t \in \mathbb{R}^3\) (IMU gyro, body frame)
  - linear acceleration \(a_t \in \mathbb{R}^3\) (IMU accel, body frame, gravity not removed)
- Exposes NumPy arrays that are ready to be windowed and fed into TensorFlow models.

## Dataset

We use the **EuRoC MAV** dataset:

- Primary info page: `http://projects.asl.ethz.ch/datasets/euroc-mav/`
- Direct research-collection entry: `https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f`

You are expected to download and extract one or more EuRoC sequences yourself, then point the preprocessing code at the local sequence directory (for example `data/raw/V1_01_easy`).

## Quick start (single sequence preprocessing)

1. **Create a virtual environment** (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download a EuRoC sequence** (e.g. `V1_01_easy`) from the ETH links above and extract it.

   A typical layout after extraction might look like:

   - `data/raw/V1_01_easy/mav0/imu0/data.csv`
   - `data/raw/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv`
   - (other EuRoC folders: `cam0`, `cam1`, etc.)

4. **Run the single-sequence preprocessing script**:

```bash
python -m mt4599.scripts.preprocess_single_sequence \
  --seq-root data/raw/V1_01_easy \
  --output data/processed/V1_01_easy_state_200hz.npz
```

This will:

- Detect IMU and pose / ground-truth CSVs under `--seq-root`.
- Resample everything to a uniform **200 Hz** grid.
- Build state vectors \(s_t\) with 16 features per timestep.
- Save a `.npz` file with arrays and metadata:
  - `state` – shape `(T, 16)`
  - `t` – timestamps in seconds, shape `(T,)`
  - `feature_names` – list of feature names
  - `meta` – small dict with additional info (sampling rate, sequence path, pose source, etc.)

## Assumptions and conventions

- **Sampling rate**: 200 Hz global rate.
  - IMU is kept at its native rate (effectively cleaned to a uniform grid).
  - Pose is interpolated up to 200 Hz.
- **Frames**:
  - IMU gyro and accel are kept in the **IMU/body frame**.
  - Position and orientation come from the EuRoC ground-truth frame (world / reference frame as defined by the dataset).
- **Pose source priority**:
  1. `mav0/state_groundtruth_estimate0/data.csv` (onboard fused estimate) when available.
  2. `vicon0/data.csv` as a fallback.
- **Gravity**:
  - Linear acceleration from the IMU **includes gravity**. Gravity compensation can be added later in the modelling or analysis stage.

## Download helper

There is a small script to help you download some standard EuRoC sequences (zip archives) directly from the ETH server.

Example: download `V1_01_easy` and `MH_01_easy` into `data/zips` **without** actually downloading (just print commands):

```bash
python -m mt4599.scripts.download_euroc \
  --output-dir data/zips \
  --sequences V1_01_easy,MH_01_easy \
  --print-only
```

On a machine where you want to download the data, you can omit `--print-only`:

```bash
python -m mt4599.scripts.download_euroc \
  --output-dir data/zips \
  --sequences V1_01_easy,MH_01_easy
```

After downloading, extract the zip files (for example into `data/raw/`), then point the preprocessing script at the extracted sequence folder (e.g. `data/raw/V1_01_easy`).

## End-to-end workflow (multi-sequence + transformer)

The following steps describe how to go from raw EuRoC zips to a trained Transformer model and embeddings on a fresh machine.

### 1. Clone repo and set up environment

```bash
git clone https://github.com/zanolablue/mt4599.git
cd mt4599

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
```

### 2. Download EuRoC sequences (example: V1_01_easy and MH_01_easy)

```bash
python -m mt4599.scripts.download_euroc \
  --output-dir data/zips \
  --sequences V1_01_easy,MH_01_easy
```

Then extract the zips into `data/raw`:

```bash
mkdir -p data/raw
unzip data/zips/V1_01_easy.zip -d data/raw
unzip data/zips/MH_01_easy.zip -d data/raw
```

You should now have directories like:

- `data/raw/V1_01_easy/mav0/imu0/data.csv`
- `data/raw/MH_01_easy/mav0/imu0/data.csv`

### 3. Preprocess multiple sequences to state vectors

```bash
python -m mt4599.scripts.preprocess_multiple_sequences \
  --input-root data/raw \
  --output-dir data/processed \
  --manifest data/processed/manifest.json
```

This runs the single-sequence preprocessing for each valid sequence subdirectory, saving:

- one `.npz` per sequence with `state (T, 16)` and timestamps, and
- a `manifest.json` describing which sequences were processed, their pose source, and basic stats.

### 4. Build a windowed dataset for next-step prediction

Use the default MVP settings \(W = 128, stride = 16\):

```bash
python -m mt4599.scripts.build_window_dataset \
  --manifest data/processed/manifest.json \
  --output data/processed/euroc_W128_S16_dataset.npz \
  --window 128 \
  --stride 16
```

This produces a single dataset `.npz` with:

- `X_train, y_train`
- `X_val, y_val`
- `X_test, y_test`
- `mu, sigma` (normalisation stats computed from training sequences only)
- metadata (`meta_json`) including sequence splits, window length, stride, and task.

### 5. Train the Transformer model

Train a baseline Transformer encoder for next-step prediction:

```bash
python -m mt4599.scripts.train_transformer \
  --dataset data/processed/euroc_W128_S16_dataset.npz \
  --output-dir runs/transformer_baseline \
  --window 128 \
  --d-model 128 --num-heads 4 --num-layers 3 --d-ff 256 \
  --dropout 0.1 --batch-size 64 --epochs 50 --learning-rate 1e-3
```

This will:

- build a small Transformer encoder with sinusoidal positional encoding,
- train it to predict the next state \(s_{t+1}\) from windows of length \(W\),
- save:
  - the prediction model (`runs/transformer_baseline/model`)
  - the encoder model (`runs/transformer_baseline/encoder`)
  - training history (`history.json`)
  - test metrics (`test_metrics.json`)
  - configuration (`config.json`)

### 6. Extract encoder embeddings for analysis

After training, extract encoder embeddings for downstream analysis (PCA, clustering, HMMs, etc.).

Example: extract embeddings for the training split:

```bash
python -m mt4599.scripts.extract_embeddings \
  --dataset data/processed/euroc_W128_S16_dataset.npz \
  --model-dir runs/transformer_baseline \
  --split train \
  --representation both \
  --output runs/transformer_baseline/embeddings_train.npz
```

This will save:

- `emb_seq`: sequence embeddings, shape `(N, W, d_model)` (per-timestep encoder states)
- `emb_last`: pooled embeddings, shape `(N, d_model)` (final-token representation)

along with a small metadata JSON describing the representation and split.

## Next steps (for the thesis project)

- Use the saved embeddings to perform PCA, k-means / GMM clustering, and HMM modelling over discrete regimes.
- Run sensitivity analyses over window length, stride, and transformer hyperparameters.
- Add lightweight plotting and sanity-check utilities (trajectory plots, IMU / pose summaries, embedding visualisations).


