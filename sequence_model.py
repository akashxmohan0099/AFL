"""
AFL Prediction Pipeline — GRU Sequence Model (Phase 4)
======================================================
Self-supervised GRU that learns player form trajectories from sequential
game-by-game data.  The pretext task is: given a player's last N games,
predict the current game's stats (GL, DI, MK).

The learned hidden state is a dense "form embedding" that captures
momentum, streaks, recent role changes, etc.  These embeddings are
extracted and merged back into the tabular feature matrix to give
tree-based models access to sequential patterns that rolling averages
cannot capture.

Requires: torch (PyTorch).  CPU-only is fine.

Usage:
    from sequence_model import build_sequences, train_form_model
    from sequence_model import extract_form_embeddings, augment_features

    sequences = build_sequences(df)
    model = train_form_model(sequences)
    form_df = extract_form_embeddings(model, sequences)
    df_aug = augment_features(df, form_df)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# PyTorch is optional — only needed for training
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GRU form model training.  "
            "Install with:  pip install torch\n"
            "CPU-only is sufficient (no GPU needed)."
        )


# ---------------------------------------------------------------------------
# Per-game stat features used as GRU input
# ---------------------------------------------------------------------------
# These are the raw stats available at each game — the GRU learns to extract
# form signals from their sequences.
SEQUENCE_STATS = [
    "GL", "DI", "MK", "KI", "HB", "HO", "TK", "FF", "FA",
]

# Targets the model predicts (self-supervised pretext)
TARGET_STATS = ["GL", "DI", "MK"]


# ---------------------------------------------------------------------------
# Build sequences from the feature matrix
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, lookback: int | None = None) -> dict:
    """Build player game sequences from the feature matrix.

    Returns dict with:
        - sequences: list of (player_id, match_id, input_seq, target) tuples
        - stat_means: array of per-stat means (for normalization)
        - stat_stds: array of per-stat stds
    """
    lookback = lookback or config.SEQUENCE_LOOKBACK

    # Validate columns exist
    required = SEQUENCE_STATS + ["player_id", "match_id", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for sequence building: {missing}")

    logger.info(f"Building sequences with lookback={lookback} from {len(df)} rows")

    # Sort by player then date for chronological ordering
    df_sorted = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # Compute normalization stats from training data
    stat_vals = df_sorted[SEQUENCE_STATS].values.astype(np.float32)
    stat_means = np.nanmean(stat_vals, axis=0)
    stat_stds = np.nanstd(stat_vals, axis=0)
    stat_stds[stat_stds < 1e-6] = 1.0  # avoid div-by-zero

    sequences = []
    for pid, grp in df_sorted.groupby("player_id", observed=True):
        vals = grp[SEQUENCE_STATS].values.astype(np.float32)
        match_ids = grp["match_id"].values

        # Normalize
        vals_norm = (vals - stat_means) / stat_stds

        # Slide window: for each game i, input is games [i-lookback : i]
        for i in range(1, len(vals_norm)):
            start = max(0, i - lookback)
            input_seq = vals_norm[start:i]  # shape (seq_len, n_stats)
            target = vals_norm[i, :len(TARGET_STATS)]  # GL, DI, MK normalized
            sequences.append((str(pid), int(match_ids[i]), input_seq, target))

    logger.info(f"Built {len(sequences)} sequences for {df_sorted['player_id'].nunique()} players")

    return {
        "sequences": sequences,
        "stat_means": stat_means,
        "stat_stds": stat_stds,
        "lookback": lookback,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset & GRU Model — only defined when torch is available
# ---------------------------------------------------------------------------

class PlayerSequenceDataset:
    """Stub replaced below if torch is available."""
    pass


class PlayerFormGRU:
    """Stub replaced below if torch is available."""
    pass


if _TORCH_AVAILABLE:
    class PlayerSequenceDataset(Dataset):  # type: ignore[no-redef]
        """Variable-length player game sequences with padding."""

        def __init__(self, sequences: list, lookback: int):
            self.sequences = sequences
            self.lookback = lookback
            self.n_features = len(SEQUENCE_STATS)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            pid, match_id, input_seq, target = self.sequences[idx]
            seq_len = len(input_seq)

            # Pad to lookback length (left-pad with zeros)
            padded = np.zeros((self.lookback, self.n_features), dtype=np.float32)
            if seq_len > 0:
                start = self.lookback - seq_len
                padded[start:] = input_seq

            # Mask: 1 where real data exists, 0 for padding
            mask = np.zeros(self.lookback, dtype=np.float32)
            if seq_len > 0:
                mask[self.lookback - seq_len:] = 1.0

            return (
                torch.from_numpy(padded),
                torch.from_numpy(mask),
                torch.tensor(target, dtype=torch.float32),
                match_id,
            )

    class PlayerFormGRU(nn.Module):  # type: ignore[no-redef]
        """GRU model for learning player form from game sequences.

        Architecture:
            Input (n_stats) → GRU (hidden_dim, n_layers) → FC → output_dim form embedding
                                                          → prediction heads (GL, DI, MK)

        The form embedding (output_dim) is the useful representation.
        The prediction heads are only for the self-supervised pretext task.
        """

        def __init__(
            self,
            n_features: int = len(SEQUENCE_STATS),
            hidden_dim: int = None,
            output_dim: int = None,
            n_layers: int = None,
        ):
            super().__init__()

            self.hidden_dim = hidden_dim or config.SEQUENCE_HIDDEN_DIM
            self.output_dim = output_dim or config.SEQUENCE_OUTPUT_DIM
            self.n_layers = n_layers or config.SEQUENCE_N_LAYERS
            self.n_features = n_features

            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=self.hidden_dim,
                num_layers=self.n_layers,
                batch_first=True,
                dropout=0.1 if self.n_layers > 1 else 0.0,
            )

            # Form embedding projection
            self.embed_fc = nn.Sequential(
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.ReLU(),
            )

            # Prediction heads (pretext task)
            self.head_gl = nn.Linear(self.output_dim, 1)
            self.head_di = nn.Linear(self.output_dim, 1)
            self.head_mk = nn.Linear(self.output_dim, 1)

        def encode(self, x, mask=None):
            """Get form embeddings without prediction heads.

            Args:
                x: (batch, seq_len, n_features) — padded input sequences
                mask: (batch, seq_len) — 1 for real data, 0 for padding

            Returns:
                (batch, output_dim) form embeddings
            """
            gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)

            # Use the last real timestep's hidden state
            if mask is not None:
                lengths = mask.sum(dim=1).long()  # (batch,)
                lengths = torch.clamp(lengths, min=1) - 1  # 0-indexed
                batch_idx = torch.arange(gru_out.size(0), device=gru_out.device)
                last_hidden = gru_out[batch_idx, lengths]  # (batch, hidden_dim)
            else:
                last_hidden = gru_out[:, -1]  # (batch, hidden_dim)

            form_emb = self.embed_fc(last_hidden)  # (batch, output_dim)
            return form_emb

        def forward(self, x, mask=None):
            """Forward pass: form embedding + pretext predictions.

            Returns:
                form_emb: (batch, output_dim)
                preds: dict with 'GL', 'DI', 'MK' predictions (batch, 1) each
            """
            form_emb = self.encode(x, mask)
            preds = {
                "GL": self.head_gl(form_emb),
                "DI": self.head_di(form_emb),
                "MK": self.head_mk(form_emb),
            }
            return form_emb, preds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_form_model(
    seq_data: dict,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    val_frac: float = 0.1,
) -> PlayerFormGRU:
    """Train the GRU form model on player sequences.

    Args:
        seq_data: output of build_sequences()
        epochs: training epochs (default from config)
        batch_size: batch size (default from config)
        lr: learning rate (default from config)
        val_frac: fraction of sequences for validation

    Returns:
        Trained PlayerFormGRU model
    """
    _require_torch()

    epochs = epochs or config.SEQUENCE_EPOCHS
    batch_size = batch_size or config.SEQUENCE_BATCH_SIZE
    lr = lr or config.SEQUENCE_LR
    lookback = seq_data["lookback"]

    sequences = seq_data["sequences"]
    n_total = len(sequences)

    # Chronological train/val split (last val_frac of sequences)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]

    logger.info(f"Training GRU: {n_train} train, {n_val} val sequences, "
                f"{epochs} epochs, batch_size={batch_size}, lr={lr}")

    train_ds = PlayerSequenceDataset(train_seqs, lookback)
    val_ds = PlayerSequenceDataset(val_seqs, lookback)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PlayerFormGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_patience = 7

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for padded, mask, target, _ in train_loader:
            optimizer.zero_grad()
            _, preds = model(padded, mask)

            loss = (
                criterion(preds["GL"].squeeze(-1), target[:, 0])
                + criterion(preds["DI"].squeeze(-1), target[:, 1])
                + criterion(preds["MK"].squeeze(-1), target[:, 2])
            ) / 3.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * padded.size(0)

        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for padded, mask, target, _ in val_loader:
                _, preds = model(padded, mask)
                loss = (
                    criterion(preds["GL"].squeeze(-1), target[:, 0])
                    + criterion(preds["DI"].squeeze(-1), target[:, 1])
                    + criterion(preds["MK"].squeeze(-1), target[:, 2])
                ) / 3.0
                val_loss += loss.item() * padded.size(0)

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} — train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"  Early stopping at epoch {epoch+1} "
                            f"(best val_loss={best_val_loss:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"GRU training complete — best val_loss={best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Extract form embeddings
# ---------------------------------------------------------------------------

def extract_form_embeddings(
    model: "PlayerFormGRU",
    seq_data: dict,
) -> pd.DataFrame:
    """Extract form embeddings for every (player, match) in the sequence data.

    Returns DataFrame with columns: player_id, match_id, form_emb_0..form_emb_{dim-1}
    """
    _require_torch()

    model.eval()
    sequences = seq_data["sequences"]
    lookback = seq_data["lookback"]
    n_features = len(SEQUENCE_STATS)

    all_pids = []
    all_mids = []
    all_embs = []

    batch_size = 2048
    dataset = PlayerSequenceDataset(sequences, lookback)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        seq_idx = 0
        for padded, mask, _, match_ids in loader:
            form_emb = model.encode(padded, mask)  # (batch, output_dim)
            embs = form_emb.cpu().numpy()

            batch_len = padded.size(0)
            for i in range(batch_len):
                pid, mid, _, _ = sequences[seq_idx + i]
                all_pids.append(pid)
                all_mids.append(mid)

            all_embs.append(embs)
            seq_idx += batch_len

    emb_matrix = np.vstack(all_embs)
    output_dim = emb_matrix.shape[1]

    emb_cols = [f"form_emb_{i}" for i in range(output_dim)]
    result = pd.DataFrame({
        "player_id": all_pids,
        "match_id": all_mids,
    })
    for i, col in enumerate(emb_cols):
        result[col] = emb_matrix[:, i].astype(np.float32)

    # Deduplicate (should be 1:1 but just in case)
    result = result.drop_duplicates(subset=["player_id", "match_id"], keep="last")

    logger.info(f"Extracted form embeddings: {result.shape[0]} rows × "
                f"{output_dim} dims")
    return result


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_form_model(model: "PlayerFormGRU", seq_data: dict, form_df: pd.DataFrame):
    """Save model, normalization stats, and form embeddings to disk."""
    _require_torch()

    out_dir = config.SEQUENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model weights
    torch.save(model.state_dict(), out_dir / "form_gru.pt")

    # Normalization stats
    np.savez(
        out_dir / "norm_stats.npz",
        stat_means=seq_data["stat_means"],
        stat_stds=seq_data["stat_stds"],
    )

    # Form embeddings
    form_df.to_parquet(out_dir / "form_embeddings.parquet", index=False)

    logger.info(f"Saved GRU model and form embeddings to {out_dir}")


def load_form_embeddings() -> pd.DataFrame | None:
    """Load pre-computed form embeddings (no torch needed)."""
    path = config.SEQUENCE_DIR / "form_embeddings.parquet"
    if not path.exists():
        logger.warning(f"No form embeddings found at {path}")
        return None
    df = pd.read_parquet(path)
    logger.info(f"Loaded form embeddings: {df.shape}")
    return df


def load_form_model() -> "PlayerFormGRU | None":
    """Load trained GRU model from disk."""
    _require_torch()

    path = config.SEQUENCE_DIR / "form_gru.pt"
    if not path.exists():
        logger.warning(f"No GRU model found at {path}")
        return None

    model = PlayerFormGRU()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    logger.info("Loaded GRU form model")
    return model


# ---------------------------------------------------------------------------
# Augment feature matrix
# ---------------------------------------------------------------------------

def augment_features(df: pd.DataFrame, form_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge form embeddings into the feature matrix.

    If form_df is None, attempts to load from disk.
    Adds form_emb_0..form_emb_{dim-1} columns.
    """
    if form_df is None:
        form_df = load_form_embeddings()

    if form_df is None:
        logger.info("No form embeddings available — skipping augmentation")
        return df

    emb_cols = [c for c in form_df.columns if c.startswith("form_emb_")]
    merge_cols = ["player_id", "match_id"] + emb_cols

    # Ensure compatible dtypes for merge
    form_merge = form_df[merge_cols].copy()
    form_merge["player_id"] = form_merge["player_id"].astype(str)
    form_merge["match_id"] = form_merge["match_id"].astype(np.int64)

    df_out = df.copy()
    df_out["player_id"] = df_out["player_id"].astype(str)
    df_out["match_id"] = df_out["match_id"].astype(np.int64)

    # Drop existing form_emb columns if re-augmenting
    existing_emb = [c for c in df_out.columns if c.startswith("form_emb_")]
    if existing_emb:
        df_out = df_out.drop(columns=existing_emb)

    before = len(df_out)
    df_out = df_out.merge(form_merge, on=["player_id", "match_id"], how="left")
    assert len(df_out) == before, "Merge changed row count — check for duplicates"

    # Fill missing embeddings (players with too few games) with zeros
    for c in emb_cols:
        df_out[c] = df_out[c].fillna(0.0).astype(np.float32)

    n_matched = df_out[emb_cols[0]].ne(0).sum()
    logger.info(f"Augmented with {len(emb_cols)} form features — "
                f"{n_matched}/{len(df_out)} rows matched ({100*n_matched/len(df_out):.1f}%)")

    return df_out
