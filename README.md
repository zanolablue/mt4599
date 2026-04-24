# mt4599

Code for my BSc thesis: *Transformer-Based Sequence Modeling for Autonomous Platforms: A Statistical Behavior Dynamics Approach* (University of St Andrews, 2026).

A transformer encoder is trained on UAV state sequences from the EuRoC MAV dataset using a next-step prediction objective. The learned latent representations are then analysed using PCA, k-means clustering, and hidden Markov modelling to recover interpretable behavioural regimes.

## Structure
mt4599/          # core package (preprocessing, models, datasets)
runs/            # training outputs (ignored by git)
requirements.txt

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download EuRoC MAV sequences from [ETH Zurich](http://projects.asl.ethz.ch/datasets/euroc-mav/):

```bash
python -m mt4599.scripts.download_euroc \
  --output-dir data/zips \
  --sequences V1_01_easy,MH_01_easy

mkdir -p data/raw
unzip data/zips/V1_01_easy.zip -d data/raw
unzip data/zips/MH_01_easy.zip -d data/raw
```

## Pipeline

```bash
# 1. Preprocess sequences to state vectors (T, 16) at 200 Hz
python -m mt4599.scripts.preprocess_multiple_sequences \
  --input-root data/raw \
  --output-dir data/processed \
  --manifest data/processed/manifest.json

# 2. Build windowed dataset for next-step prediction
python -m mt4599.scripts.build_window_dataset \
  --manifest data/processed/manifest.json \
  --output data/processed/euroc_W128_S16_dataset.npz \
  --window 128 --stride 16

# 3. Train transformer encoder
python -m mt4599.scripts.train_transformer \
  --dataset data/processed/euroc_W128_S16_dataset.npz \
  --output-dir runs/transformer_baseline \
  --window 128 --d-model 128 --num-heads 4 --num-layers 3 \
  --d-ff 256 --dropout 0.1 --epochs 50

# 4. Extract embeddings for analysis
python -m mt4599.scripts.extract_embeddings \
  --dataset data/processed/euroc_W128_S16_dataset.npz \
  --model-dir runs/transformer_baseline \
  --split all --representation both \
  --output runs/transformer_baseline/embeddings.npz
```

## State vector

`s_t = [Δp, v, q, ω, a]` — 16-dimensional, 200 Hz. Position increments, Savitzky–Golay velocity, gravity-removed world-frame acceleration, quaternion orientation, body-frame angular velocity.


