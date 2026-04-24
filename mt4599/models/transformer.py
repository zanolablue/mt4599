from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


@dataclass
class TransformerConfig:
    window_length: int
    feature_dim: int = 16
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1

    def to_dict(self) -> Dict:
        return asdict(self)


def _get_sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Create standard sinusoidal positional encodings as in Vaswani et al. 2017.

    Returns:
        (1, max_len, d_model) array.
    """
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model)
    )
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[np.newaxis, ...]


class LastTimestep(layers.Layer):
    """Extract the last timestep from a sequence: (batch, T, D) -> (batch, D).

    Replaces a Lambda layer so the model is fully serialisable with Keras 3.
    """

    def call(self, x):
        return x[:, -1, :]

    def get_config(self):
        return super().get_config()


def build_transformer_models(config: TransformerConfig) -> Tuple[Model, Model]:
    """
    Build a baseline Transformer encoder for next-step prediction and a separate encoder model.

    - Input: (batch, window_length, feature_dim)
    - Output (prediction model): (batch, feature_dim) next-step prediction.
    - Output (encoder model): (batch, window_length, d_model) encoder hidden states.
    """
    W = config.window_length
    D = config.feature_dim
    inputs = layers.Input(shape=(W, D), name="input_sequence")

    # Linear projection to d_model
    x = layers.Dense(config.d_model, name="input_projection")(inputs)

    # Sinusoidal positional encoding
    pe = _get_sinusoidal_positional_encoding(W, config.d_model)
    pe_const = tf.constant(pe, dtype=tf.float32, name="positional_encoding")
    x = x + pe_const[:, :W, :]

    # Encoder blocks
    for i in range(config.num_layers):
        attn_name = f"encoder_mha_{i}"
        ffn_name = f"encoder_ffn_{i}"

        # Self-attention block
        x_norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        attn_out = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.d_model // config.num_heads,
            dropout=config.dropout,
            name=attn_name,
        )(x_norm1, x_norm1)
        attn_out = layers.Dropout(config.dropout, name=f"dropout_attn_{i}")(attn_out)
        x = layers.Add(name=f"residual_attn_{i}")([x, attn_out])

        # Feed-forward block
        x_norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        ffn_inner = layers.Dense(config.d_ff, activation="relu", name=f"{ffn_name}_inner")(
            x_norm2
        )
        ffn_out = layers.Dense(config.d_model, name=f"{ffn_name}_out")(ffn_inner)
        ffn_out = layers.Dropout(config.dropout, name=f"dropout_ffn_{i}")(ffn_out)
        x = layers.Add(name=f"residual_ffn_{i}")([x, ffn_out])

    encoder_outputs = layers.LayerNormalization(epsilon=1e-6, name="encoder_output_ln")(x)

    # Last-timestep representation for next-step prediction.
    # Uses a proper named Layer subclass instead of Lambda — Lambda is not
    # safely serialisable in Keras 3 and will crash model.save().
    last_token = LastTimestep(name="last_token")(encoder_outputs)
    predictions = layers.Dense(D, name="prediction_head")(last_token)

    model = Model(inputs=inputs, outputs=predictions, name="euroc_transformer_next_step")
    encoder = Model(inputs=inputs, outputs=encoder_outputs, name="euroc_transformer_encoder")

    return model, encoder
