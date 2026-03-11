"""
AFL Prediction Pipeline — Entity Embedding Module
===================================================
Learns dense vector representations for categorical entities (player_id,
team, opponent, venue, archetype) via a multi-task neural network trained
on goals, disposals, and marks prediction jointly.

The learned embeddings can be extracted and merged back into the tabular
feature matrix to give tree-based models access to entity similarity
information that one-hot encoding cannot capture.

Requires: torch (PyTorch).  Install with ``pip install torch`` (CPU-only
is fine).  The load_embeddings() and augment_features() functions work
without torch — they only need pandas and parquet files on disk.

Usage:
    from embeddings import train_embedding_model, extract_embeddings
    from embeddings import save_embeddings, load_embeddings, augment_features

    model, vocabs = train_embedding_model(df, feature_cols)
    emb_dfs = extract_embeddings(model, vocabs)
    save_embeddings(model, vocabs, emb_dfs)

    # Later — augment feature matrix with pre-computed embeddings
    df_aug = augment_features(df)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

# PyTorch is an optional dependency — required only for training / model
# operations, NOT for loading pre-computed parquet embeddings.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    """Raise a clear error when torch is needed but not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for entity embedding training and model "
            "operations.  Install it with:  pip install torch\n"
            "CPU-only is sufficient (no GPU needed)."
        )


# ---------------------------------------------------------------------------
# Vocabulary mapping (no torch dependency)
# ---------------------------------------------------------------------------

class EntityVocab:
    """Maps string entity values to integer indices. Index 0 = <UNK>."""

    def __init__(self):
        self.str2idx: dict[str, int] = {"<UNK>": 0}
        self.idx2str: dict[int, str] = {0: "<UNK>"}

    def fit(self, values):
        """Build vocab from array of string values."""
        for v in sorted(set(str(x) for x in values if pd.notna(x))):
            if v not in self.str2idx:
                idx = len(self.str2idx)
                self.str2idx[v] = idx
                self.idx2str[idx] = v
        return self

    def transform(self, values):
        """Convert string values to integer indices."""
        return np.array([self.str2idx.get(str(v), 0) for v in values], dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.str2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.str2idx, f, indent=2)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path) as f:
            vocab.str2idx = json.load(f)
        vocab.idx2str = {int(v): k for k, v in vocab.str2idx.items()}
        return vocab

    def __repr__(self):
        return f"EntityVocab(size={self.size})"


# ---------------------------------------------------------------------------
# Neural network (requires torch)
# ---------------------------------------------------------------------------

class EntityEmbeddingNet:
    """Multi-task entity embedding network.

    Learns dense representations for categorical entities by training
    on goals, disposals, and marks prediction jointly.

    This is a wrapper that defines the actual nn.Module at construction
    time, so the class itself can be referenced without torch installed.
    The constructor will raise ImportError if torch is missing.
    """
    pass  # Replaced below if torch is available


if _TORCH_AVAILABLE:
    class EntityEmbeddingNet(nn.Module):  # type: ignore[no-redef]
        """Multi-task entity embedding network.

        Learns dense representations for categorical entities by training
        on goals, disposals, and marks prediction jointly.
        """

        def __init__(self, vocab_sizes, embedding_dims, n_numeric, hidden_dim=128):
            """
            Args:
                vocab_sizes:    dict  e.g. {"player_id": 1500, "team": 20, ...}
                embedding_dims: dict  e.g. {"player_id": 32,   "team": 8,  ...}
                n_numeric:      int   number of continuous feature columns
                hidden_dim:     int   width of the first hidden layer
            """
            super().__init__()
            self.entity_names = sorted(vocab_sizes.keys())

            self.embeddings = nn.ModuleDict({
                name: nn.Embedding(vocab_sizes[name], embedding_dims[name], padding_idx=0)
                for name in self.entity_names
            })

            total_emb_dim = sum(embedding_dims[name] for name in self.entity_names)

            self.numeric_bn = nn.BatchNorm1d(n_numeric)
            input_dim = total_emb_dim + n_numeric

            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # Multi-task prediction heads (ReLU ensures non-negative counts)
            self.head_gl = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.ReLU())
            self.head_di = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.ReLU())
            self.head_mk = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.ReLU())

        def forward(self, entity_indices, numeric_features):
            """
            Args:
                entity_indices:   dict of {name: LongTensor[batch]}
                numeric_features: FloatTensor[batch, n_numeric]

            Returns:
                dict with keys "GL", "DI", "MK" — each FloatTensor[batch]
            """
            emb_parts = [self.embeddings[name](entity_indices[name])
                          for name in self.entity_names]
            emb_cat = torch.cat(emb_parts, dim=1)

            num_normed = self.numeric_bn(numeric_features)
            x = torch.cat([emb_cat, num_normed], dim=1)
            h = self.trunk(x)

            return {
                "GL": self.head_gl(h).squeeze(-1),
                "DI": self.head_di(h).squeeze(-1),
                "MK": self.head_mk(h).squeeze(-1),
            }

        def get_embeddings(self, entity_name):
            """Extract learned embedding weights as numpy array.

            Returns:
                np.ndarray of shape (vocab_size, embedding_dim)
            """
            return self.embeddings[entity_name].weight.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Entity columns in order of expected appearance in the feature matrix.
ENTITY_COLS = ["player_id", "team", "opponent", "venue", "archetype"]

# Target columns and their loss weights (disposals have larger variance,
# so we down-weight them relative to goals/marks).
TARGET_COLS = ["GL", "DI", "MK"]
TARGET_LOSS_WEIGHTS = {"GL": 1.0, "DI": 0.3, "MK": 0.5}


def _build_vocabs(df):
    """Build EntityVocab for each entity column present in *df*."""
    vocabs = {}
    for col in ENTITY_COLS:
        if col not in df.columns:
            continue
        if col == "archetype":
            vocabs[col] = EntityVocab().fit(df[col].astype(str).values)
        else:
            vocabs[col] = EntityVocab().fit(df[col].values)
    return vocabs


def _prepare_tensors(df, vocabs, feature_cols):
    """Convert a DataFrame into tensors suitable for the embedding net.

    Returns:
        entity_tensors: dict {name: LongTensor}
        numeric_tensor: FloatTensor
        target_tensors: dict {target: FloatTensor}
        weight_tensor:  FloatTensor (sample weights, all-ones if absent)
    """
    _require_torch()

    # Entity indices
    entity_tensors = {}
    for name, vocab in vocabs.items():
        if name == "archetype":
            entity_tensors[name] = torch.from_numpy(
                vocab.transform(df["archetype"].astype(str).values)
            )
        else:
            entity_tensors[name] = torch.from_numpy(
                vocab.transform(df[name].values)
            )

    # Numeric features — fill NaN with 0 for the neural net
    available_cols = [c for c in feature_cols if c in df.columns]
    numeric = df[available_cols].values.astype(np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    numeric_tensor = torch.from_numpy(numeric)

    # Targets
    target_tensors = {}
    for t in TARGET_COLS:
        if t in df.columns:
            vals = df[t].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            target_tensors[t] = torch.from_numpy(vals)
        else:
            target_tensors[t] = torch.zeros(len(df), dtype=torch.float32)

    # Sample weights
    if "sample_weight" in df.columns:
        w = df["sample_weight"].values.astype(np.float32)
        w = np.nan_to_num(w, nan=1.0)
        weight_tensor = torch.from_numpy(w)
    else:
        weight_tensor = torch.ones(len(df), dtype=torch.float32)

    return entity_tensors, numeric_tensor, target_tensors, weight_tensor


def train_embedding_model(df, feature_cols, epochs=None, batch_size=None, lr=None):
    """Train the entity embedding network.

    Args:
        df:           Feature matrix with entity columns (player_id, team,
                      opponent, venue, archetype), numeric feature columns,
                      and target columns (GL, DI, MK).
        feature_cols: list of numeric feature column names.
        epochs:       Number of training epochs (default from config).
        batch_size:   Mini-batch size (default from config).
        lr:           Learning rate (default from config).

    Returns:
        (model, vocabs)  — trained EntityEmbeddingNet and dict of EntityVocab
    """
    _require_torch()

    epochs = epochs or config.EMBEDDING_EPOCHS
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    lr = lr or config.EMBEDDING_LR
    hidden_dim = config.EMBEDDING_HIDDEN_DIM
    embedding_dims = dict(config.EMBEDDING_DIMS)

    # ── 1. Build vocabularies ────────────────────────────────────────────
    vocabs = _build_vocabs(df)

    vocab_sizes = {name: vocab.size for name, vocab in vocabs.items()}
    # Only keep embedding dims for entities that are actually present
    embedding_dims = {name: embedding_dims.get(name, 8) for name in vocabs}

    available_feature_cols = [c for c in feature_cols if c in df.columns]
    n_numeric = len(available_feature_cols)

    print(f"Entity embedding training:")
    for name, vocab in vocabs.items():
        print(f"  {name:12s}: vocab_size={vocab.size:>5d}, emb_dim={embedding_dims[name]}")
    print(f"  numeric features: {n_numeric}")
    print(f"  dataset rows:     {len(df)}")

    # ── 2. Prepare tensors ───────────────────────────────────────────────
    entity_tensors, numeric_tensor, target_tensors, weight_tensor = \
        _prepare_tensors(df, vocabs, available_feature_cols)

    # ── 3. Train / val split (last 10% by date) ─────────────────────────
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        sorted_dates = dates.sort_values()
        cutoff = sorted_dates.iloc[int(len(sorted_dates) * 0.9)]
        train_mask = dates <= cutoff
        val_mask = ~train_mask
    else:
        n = len(df)
        split = int(n * 0.9)
        train_mask = pd.Series([True] * split + [False] * (n - split))
        val_mask = ~train_mask

    train_idx = torch.where(torch.from_numpy(train_mask.values))[0]
    val_idx = torch.where(torch.from_numpy(val_mask.values))[0]

    print(f"  train samples:    {len(train_idx)}")
    print(f"  val samples:      {len(val_idx)}")

    # ── 4. Build DataLoaders ─────────────────────────────────────────────
    # Pack all tensors into a flat list for TensorDataset:
    # [entity_0, entity_1, ..., numeric, GL, DI, MK, weight]
    entity_names_sorted = sorted(vocabs.keys())

    def _make_dataset(idx):
        parts = []
        for name in entity_names_sorted:
            parts.append(entity_tensors[name][idx])
        parts.append(numeric_tensor[idx])
        for t in TARGET_COLS:
            parts.append(target_tensors[t][idx])
        parts.append(weight_tensor[idx])
        return TensorDataset(*parts)

    train_ds = _make_dataset(train_idx)
    val_ds = _make_dataset(val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            drop_last=False)

    n_entity = len(entity_names_sorted)

    def _unpack_batch(batch):
        """Unpack a flat TensorDataset batch back into structured dicts."""
        ent = {entity_names_sorted[i]: batch[i] for i in range(n_entity)}
        numeric = batch[n_entity]
        targets = {TARGET_COLS[i]: batch[n_entity + 1 + i] for i in range(len(TARGET_COLS))}
        weights = batch[n_entity + 1 + len(TARGET_COLS)]
        return ent, numeric, targets, weights

    # ── 5. Instantiate model + optimizer + scheduler ─────────────────────
    model = EntityEmbeddingNet(vocab_sizes, embedding_dims, n_numeric, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss(reduction="none")

    # ── 6. Training loop with early stopping ─────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            ent, numeric, targets, weights = _unpack_batch(batch)
            preds = model(ent, numeric)

            loss = torch.zeros(1)
            for t in TARGET_COLS:
                per_sample = mse(preds[t], targets[t])  # [batch]
                weighted = (per_sample * weights).sum() / weights.sum()
                loss = loss + TARGET_LOSS_WEIGHTS[t] * weighted

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(weights)
            train_n += len(weights)

        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                ent, numeric, targets, weights = _unpack_batch(batch)
                preds = model(ent, numeric)

                loss = torch.zeros(1)
                for t in TARGET_COLS:
                    per_sample = mse(preds[t], targets[t])
                    weighted = (per_sample * weights).sum() / weights.sum()
                    loss = loss + TARGET_LOSS_WEIGHTS[t] * weighted

                val_loss_sum += loss.item() * len(weights)
                val_n += len(weights)

        train_avg = train_loss_sum / max(train_n, 1)
        val_avg = val_loss_sum / max(val_n, 1)
        current_lr = scheduler.get_last_lr()[0]

        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_avg:.4f}  val_loss={val_avg:.4f}  "
              f"lr={current_lr:.2e}", flush=True)

        # ── Early stopping ───────────────────────────────────────────────
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_loss={best_val_loss:.4f})")

    model.eval()
    return model, vocabs


# ---------------------------------------------------------------------------
# Extraction & persistence
# ---------------------------------------------------------------------------

def extract_embeddings(model, vocabs):
    """Extract learned embeddings as DataFrames.

    Returns:
        dict: {"player_id": DataFrame[entity, emb_player_id_0, ...], ...}
    """
    _require_torch()

    result = {}
    for name, vocab in vocabs.items():
        weights = model.get_embeddings(name)
        dim = weights.shape[1]
        records = []
        for idx, entity in vocab.idx2str.items():
            if entity == "<UNK>":
                continue
            record = {"entity": entity}
            for d in range(dim):
                record[f"emb_{name}_{d}"] = float(weights[idx, d])
            records.append(record)
        result[name] = pd.DataFrame(records)
    return result


def save_embeddings(model, vocabs, embedding_dfs, path=None):
    """Save model weights, vocabularies, and extracted embedding DataFrames.

    Directory layout under *path*:
        embedding_model.pt           — PyTorch state dict
        embedding_meta.json          — architecture metadata (n_numeric, hidden_dim)
        vocab_{entity}.json          — str2idx mapping
        {entity}_embeddings.parquet  — extracted vectors
    """
    _require_torch()

    path = Path(path or config.EMBEDDINGS_DIR)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "embedding_model.pt")

    # Save architecture metadata so load_model can reconstruct without guessing
    embedding_dims = {name: config.EMBEDDING_DIMS.get(name, 8) for name in vocabs}
    total_emb = sum(embedding_dims[n] for n in sorted(vocabs.keys()))
    input_dim = model.trunk[0].in_features
    n_numeric = input_dim - total_emb
    hidden_dim = model.trunk[0].out_features
    meta = {"n_numeric": n_numeric, "hidden_dim": hidden_dim}
    with open(path / "embedding_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    for name, vocab in vocabs.items():
        vocab.save(path / f"vocab_{name}.json")
    for name, edf in embedding_dfs.items():
        edf.to_parquet(path / f"{name}_embeddings.parquet", index=False)

    print(f"Saved embeddings to {path}")


def load_embeddings(path=None):
    """Load pre-computed embedding DataFrames.

    Does NOT require torch — reads parquet files only.

    Returns:
        dict: {"player_id": DataFrame, "team": DataFrame, ...}
    """
    path = Path(path or config.EMBEDDINGS_DIR)
    result = {}
    for p in sorted(path.glob("*_embeddings.parquet")):
        name = p.stem.replace("_embeddings", "")
        result[name] = pd.read_parquet(p)
    return result


def load_model(path=None, vocabs=None):
    """Reload a saved EntityEmbeddingNet from disk.

    If *vocabs* is None, they are loaded from the same directory.

    Returns:
        (model, vocabs)
    """
    _require_torch()

    path = Path(path or config.EMBEDDINGS_DIR)

    if vocabs is None:
        vocabs = {}
        for vp in sorted(path.glob("vocab_*.json")):
            name = vp.stem.replace("vocab_", "")
            vocabs[name] = EntityVocab.load(vp)

    embedding_dims = {name: config.EMBEDDING_DIMS.get(name, 8) for name in vocabs}
    vocab_sizes = {name: vocab.size for name, vocab in vocabs.items()}

    # We need n_numeric to reconstruct the architecture. Store it in a
    # metadata file; fall back to probing the saved state dict.
    meta_path = path / "embedding_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        n_numeric = meta["n_numeric"]
        hidden_dim = meta.get("hidden_dim", config.EMBEDDING_HIDDEN_DIM)
    else:
        # Probe from weight shapes: trunk.0.weight has shape (hidden, total_emb + n_numeric)
        state = torch.load(path / "embedding_model.pt", map_location="cpu",
                           weights_only=True)
        total_emb = sum(embedding_dims[n] for n in sorted(vocabs.keys()))
        input_dim = state["trunk.0.weight"].shape[1]
        n_numeric = input_dim - total_emb
        hidden_dim = state["trunk.0.weight"].shape[0]

    model = EntityEmbeddingNet(vocab_sizes, embedding_dims, n_numeric, hidden_dim)
    model.load_state_dict(
        torch.load(path / "embedding_model.pt", map_location="cpu",
                    weights_only=True)
    )
    model.eval()
    return model, vocabs


# ---------------------------------------------------------------------------
# Feature augmentation (no torch dependency)
# ---------------------------------------------------------------------------

def augment_features(df, embedding_dfs=None):
    """Join embedding vectors onto the feature matrix.

    For each entity type, merges the learned embedding columns by matching
    entity values.  Unknown entities get zero vectors.

    Does NOT require torch — works with pre-computed parquet embeddings.

    Args:
        df:            Feature matrix DataFrame.
        embedding_dfs: dict of {entity_name: DataFrame} as returned by
                       extract_embeddings().  If None, loads from disk.

    Returns:
        DataFrame with additional ``emb_{entity}_{d}`` columns.
    """
    if embedding_dfs is None:
        embedding_dfs = load_embeddings()

    if not embedding_dfs:
        print("Warning: no embedding DataFrames found — returning df unchanged")
        return df

    result = df.copy()

    for name, emb_df in embedding_dfs.items():
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        if not emb_cols:
            continue

        if name == "archetype":
            merge_emb = emb_df.copy()
            # archetype in the feature matrix is an int; entity column is str
            merge_emb["archetype"] = merge_emb["entity"].astype(int)
            result = result.merge(
                merge_emb[["archetype"] + emb_cols],
                on="archetype", how="left",
            )
        elif name in result.columns:
            merge_emb = emb_df.rename(columns={"entity": name})
            # Ensure matching dtypes (categories need str)
            if hasattr(result[name], "cat"):
                merge_emb[name] = merge_emb[name].astype(result[name].dtype)
            result = result.merge(
                merge_emb[[name] + emb_cols],
                on=name, how="left",
            )

    # Fill NaN embeddings (unseen entities) with 0
    all_emb_cols = [c for c in result.columns if c.startswith("emb_")]
    if all_emb_cols:
        result[all_emb_cols] = result[all_emb_cols].fillna(0).astype("float32")

    return result


# ---------------------------------------------------------------------------
# Quick sanity check when run as script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("embeddings.py — smoke test")

    if not _TORCH_AVAILABLE:
        print("torch not installed — testing non-torch components only")

        # Test EntityVocab
        v = EntityVocab()
        v.fit(["a", "b", "c", "a"])
        assert v.size == 4  # <UNK> + a, b, c
        indices = v.transform(["a", "c", "z"])
        assert indices[0] == v.str2idx["a"]
        assert indices[2] == 0  # unknown -> <UNK>

        # Test augment_features with empty embeddings
        fake = pd.DataFrame({"team": ["A", "B"], "x": [1.0, 2.0]})
        aug = augment_features(fake, {})
        assert len(aug) == 2

        # Test augment_features with mock embedding dfs
        emb_team = pd.DataFrame({
            "entity": ["A", "B"],
            "emb_team_0": [0.1, 0.2],
            "emb_team_1": [0.3, 0.4],
        })
        aug = augment_features(fake, {"team": emb_team})
        assert "emb_team_0" in aug.columns
        assert aug["emb_team_0"].iloc[0] == 0.1

        print("EntityVocab and augment_features: OK")
        print("OK (torch-free subset)")
    else:
        # Full smoke test with torch
        n = 500
        rng = np.random.RandomState(config.RANDOM_SEED)
        fake = pd.DataFrame({
            "player_id": rng.choice(["p1", "p2", "p3", "p4", "p5"], n),
            "team": rng.choice(["TeamA", "TeamB", "TeamC"], n),
            "opponent": rng.choice(["TeamA", "TeamB", "TeamC"], n),
            "venue": rng.choice(["MCG", "SCG", "Gabba"], n),
            "archetype": rng.choice([0, 1, 2, 3], n),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feat_1": rng.randn(n).astype(np.float32),
            "feat_2": rng.randn(n).astype(np.float32),
            "feat_3": rng.randn(n).astype(np.float32),
            "GL": rng.poisson(1.0, n).astype(np.float32),
            "DI": rng.poisson(18.0, n).astype(np.float32),
            "MK": rng.poisson(4.0, n).astype(np.float32),
        })

        feature_cols = ["feat_1", "feat_2", "feat_3"]

        model, vocabs = train_embedding_model(fake, feature_cols, epochs=5, batch_size=64)
        emb_dfs = extract_embeddings(model, vocabs)

        for name, edf in emb_dfs.items():
            print(f"  {name}: {edf.shape}")

        aug = augment_features(fake, emb_dfs)
        new_cols = [c for c in aug.columns if c.startswith("emb_")]
        print(f"  augmented columns: {len(new_cols)}")
        print("OK")
