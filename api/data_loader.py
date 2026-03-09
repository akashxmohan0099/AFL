"""
Data loader — singleton cache for all parquet data.
Loaded once at FastAPI startup via lifespan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rapidfuzz import fuzz, process

# Add project root to path so we can import config, store, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from store import LearningStore


class DataCache:
    """In-memory cache of all parquet data and search indexes."""

    _instance = None

    def __init__(self):
        self.player_games: pd.DataFrame = pd.DataFrame()
        self.matches: pd.DataFrame = pd.DataFrame()
        self.team_matches: pd.DataFrame = pd.DataFrame()
        self.odds: pd.DataFrame = pd.DataFrame()
        self.player_odds: pd.DataFrame = pd.DataFrame()
        self.footywire: pd.DataFrame = pd.DataFrame()
        self.weather: pd.DataFrame = pd.DataFrame()
        self.umpires: pd.DataFrame = pd.DataFrame()
        self.coaches: pd.DataFrame = pd.DataFrame()
        self.experiments: List[dict] = []
        self.store: Optional[LearningStore] = None
        self.sequential_store: Optional[LearningStore] = None

        # Search index: normalized_name -> list of player_ids
        self._player_index: Dict[str, List[str]] = {}
        # player_id -> display name
        self._player_names: Dict[str, str] = {}
        self.is_loaded = False

    @classmethod
    def get(cls) -> "DataCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self):
        base = config.BASE_STORE_DIR
        self.is_loaded = False

        self.player_games = self._load(base / "player_games.parquet")
        self.matches = self._load(base / "matches.parquet")
        self.team_matches = self._load(base / "team_matches.parquet")
        self.odds = self._load(base / "odds.parquet")
        self.player_odds = self._load(base / "player_odds.parquet")
        self.footywire = self._load(base / "footywire_advanced.parquet")
        self.weather = self._load(base / "weather.parquet")
        self.umpires = self._load(base / "umpires.parquet")
        self.coaches = self._load(base / "coaches.parquet")

        self.store = LearningStore(base_dir=config.LEARNING_DIR)
        self.sequential_store = LearningStore(base_dir=config.SEQUENTIAL_DIR)
        self._load_experiments()
        self._player_index = {}
        self._player_names = {}
        self._build_player_index()
        self.is_loaded = True

        rows = len(self.player_games)
        matches = len(self.matches)
        print(f"DataCache loaded: {rows:,} player-games, {matches:,} matches, "
              f"{len(self._player_index):,} unique players")

    def _load(self, path: Path) -> pd.DataFrame:
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                print(f"WARNING: Failed to load {path}: {e}")
                return pd.DataFrame()
        print(f"  Warning: {path} not found")
        return pd.DataFrame()

    def _load_experiments(self):
        exp_dir = config.EXPERIMENTS_DIR
        if not exp_dir.exists():
            return
        self.experiments = []
        for f in sorted(exp_dir.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
            data["_filename"] = f.name
            self.experiments.append(data)

    def _build_player_index(self):
        if self.player_games.empty:
            return

        pg = self.player_games
        # Group by player_id, get the most recent name
        latest = (
            pg.sort_values("date")
            .groupby("player_id", observed=True)
            .agg(player=("player", "last"), team=("team", "last"))
        )

        for pid, row in latest.iterrows():
            name = row["player"]
            self._player_names[pid] = name
            # Normalize: "Cripps, Patrick" -> "patrick cripps"
            parts = str(name).split(", ", 1)
            if len(parts) == 2:
                normalized = f"{parts[1]} {parts[0]}".lower().strip()
            else:
                normalized = str(name).lower().strip()

            self._player_index.setdefault(normalized, []).append(pid)

    def search_players(self, query: str, limit: int = 20) -> list[dict]:
        if not query or not self._player_index:
            return []

        q = query.lower().strip()
        names = list(self._player_index.keys())
        results = process.extract(q, names, scorer=fuzz.WRatio, limit=limit)

        out = []
        seen = set()
        for name, score, _ in results:
            if score < 50:
                continue
            for pid in self._player_index[name]:
                if pid in seen:
                    continue
                seen.add(pid)
                display_name = self._player_names.get(pid, name)
                # Get latest team
                mask = self.player_games["player_id"] == pid
                player_data = self.player_games.loc[mask]
                team = player_data["team"].iloc[-1] if len(player_data) > 0 else ""
                total_games = len(player_data)
                out.append({
                    "player_id": pid,
                    "name": display_name,
                    "team": team,
                    "total_games": total_games,
                    "score": score,
                })
        return sorted(out, key=lambda x: -x["score"])[:limit]

    def get_player_name(self, player_id: str) -> str:
        return self._player_names.get(player_id, player_id)
