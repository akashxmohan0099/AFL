import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import analysis
import pipeline
from api.services import metrics_service, odds_service, player_service, season_service
from store import LearningStore


class _FakeStore:
    def __init__(self, predictions=None, outcomes=None, game_predictions=None, rounds=None):
        self._predictions = predictions or {}
        self._outcomes = outcomes or {}
        self._game_predictions = game_predictions or {}
        self._rounds = rounds or []

    def load_predictions(self, year=None, round_num=None, **kwargs):
        return self._predictions.get((year, round_num), pd.DataFrame())

    def load_outcomes(self, year=None, round_num=None, **kwargs):
        return self._outcomes.get((year, round_num), pd.DataFrame())

    def load_game_predictions(self, year=None, round_num=None, **kwargs):
        return self._game_predictions.get((year, round_num), pd.DataFrame())

    def list_rounds(self, subdir="predictions", year=None, **kwargs):
        return list(self._rounds)


class TestLearningStoreLatestLoad(unittest.TestCase):
    def test_latest_year_load_prefers_run_files_over_legacy_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LearningStore(base_dir=tmpdir)

            run_df = pd.DataFrame([{"player": "Run Player", "team": "A", "match_id": 1}])
            legacy_df = pd.DataFrame([{"player": "Legacy Player", "team": "A", "match_id": 1}])

            store.save_predictions(2025, 1, run_df, run_id="unit")
            legacy_path = Path(tmpdir) / "predictions" / "2025_R01.parquet"
            legacy_df.to_parquet(legacy_path, index=False)

            loaded = store.load_predictions(year=2025)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0]["player"], "Run Player")


class TestOddsService(unittest.TestCase):
    def test_first_goal_market_does_not_expose_anytime_scorer_probability(self):
        predictions = {
            (2025, 1): pd.DataFrame(
                [
                    {
                        "player": "Player, One",
                        "team": "Team A",
                        "match_id": 10,
                        "p_scorer": 0.91,
                    }
                ]
            )
        }
        fake_cache = SimpleNamespace(
            matches=pd.DataFrame([{"year": 2025, "round_number": 1, "match_id": 10}]),
            player_odds=pd.DataFrame(
                [
                    {
                        "match_id": 10,
                        "player": "Player, One",
                        "market_fgs_price": 5.0,
                        "market_fgs_implied_prob": 0.2,
                    }
                ]
            ),
            sequential_store=_FakeStore(predictions=predictions),
            store=None,
        )

        with patch("api.services.odds_service.DataCache.get", return_value=fake_cache):
            rows = odds_service.get_player_odds(2025, 1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_type"], "first_goal")
        self.assertEqual(rows[0]["team"], "Team A")
        self.assertNotIn("model_prob", rows[0])
        self.assertNotIn("edge", rows[0])

    def test_player_odds_match_predictions_by_match_id(self):
        predictions = {
            (2025, 1): pd.DataFrame(
                [
                    {
                        "player": "Smith, Tom",
                        "team": "Team A",
                        "match_id": 10,
                        "p_20plus_disp": 0.74,
                    },
                    {
                        "player": "Smith, Tom",
                        "team": "Team B",
                        "match_id": 11,
                        "p_20plus_disp": 0.11,
                    },
                ]
            )
        }
        fake_cache = SimpleNamespace(
            matches=pd.DataFrame([{"year": 2025, "round_number": 1, "match_id": 10}]),
            player_odds=pd.DataFrame(
                [
                    {
                        "match_id": 10,
                        "player": "Smith, Tom",
                        "market_disposal_line": 20.0,
                        "market_disposal_over_price": 1.9,
                        "market_disposal_implied_over": 0.52,
                    }
                ]
            ),
            sequential_store=_FakeStore(predictions=predictions),
            store=None,
        )

        with patch("api.services.odds_service.DataCache.get", return_value=fake_cache):
            rows = odds_service.get_player_odds(2025, 1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "Team A")
        self.assertEqual(rows[0]["model_prob"], 0.74)

    def test_game_odds_uses_canonical_store(self):
        game_predictions = {
            (2025, 1): pd.DataFrame(
                [
                    {
                        "match_id": 10,
                        "home_team": "Team A",
                        "away_team": "Team B",
                        "home_win_prob": 0.61,
                    }
                ]
            )
        }
        fake_cache = SimpleNamespace(
            matches=pd.DataFrame(
                [
                    {
                        "year": 2025,
                        "round_number": 1,
                        "match_id": 10,
                        "home_team": "Team A",
                        "away_team": "Team B",
                    }
                ]
            ),
            odds=pd.DataFrame(
                [{"match_id": 10, "home_implied_prob": 0.55, "away_implied_prob": 0.45}]
            ),
            sequential_store=_FakeStore(game_predictions=game_predictions),
            store=_FakeStore(),
        )

        with patch("api.services.odds_service.DataCache.get", return_value=fake_cache):
            rows = odds_service.get_game_odds(2025, 1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_home_prob"], 0.61)
        self.assertEqual(rows[0]["edge_home"], 0.06)


class TestPlayerService(unittest.TestCase):
    def test_get_player_predictions_filters_to_team_hint(self):
        predictions = {
            (2025, None): pd.DataFrame(
                [
                    {
                        "player": "Williams, Bailey",
                        "team": "West Coast",
                        "match_id": 1,
                        "predicted_goals": 0.2,
                    },
                    {
                        "player": "Williams, Bailey",
                        "team": "Western Bulldogs",
                        "match_id": 2,
                        "predicted_goals": 0.8,
                    },
                ]
            )
        }
        fake_cache = SimpleNamespace(
            sequential_store=_FakeStore(predictions=predictions),
            store=None,
            get_player_name=lambda player_id: "Williams, Bailey",
        )

        with patch("api.services.player_service.DataCache.get", return_value=fake_cache):
            rows = player_service.get_player_predictions("Williams, Bailey_West Coast", year=2025)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "West Coast")


class TestMetricsService(unittest.TestCase):
    def test_backtest_metrics_uses_sequential_store_and_round_level_loads(self):
        seq_preds = {
            (2025, 1): pd.DataFrame(
                [
                    {
                        "player": "Player, One",
                        "team": "Team A",
                        "match_id": 10,
                        "predicted_goals": 1.2,
                        "predicted_disposals": 18.0,
                        "predicted_marks": 5.0,
                        "p_scorer": 0.7,
                        "p_2plus_goals": 0.3,
                        "p_20plus_disp": 0.4,
                        "p_5plus_mk": 0.6,
                    }
                ]
            )
        }
        seq_outcomes = {
            (2025, 1): pd.DataFrame(
                [
                    {
                        "player": "Player, One",
                        "team": "Team A",
                        "match_id": 10,
                        "actual_goals": 1,
                        "actual_disposals": 20,
                        "actual_marks": 4,
                    }
                ]
            )
        }
        fake_cache = SimpleNamespace(
            sequential_store=_FakeStore(
                predictions=seq_preds,
                outcomes=seq_outcomes,
                rounds=[(2025, 1)],
            ),
            store=_FakeStore(),
        )

        with patch("api.services.metrics_service.DataCache.get", return_value=fake_cache):
            result = metrics_service.get_backtest_metrics(2025)

        self.assertNotIn("error", result)
        self.assertEqual(result["year"], 2025)
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(result["n_predictions"], 1)


class TestSeasonService(unittest.TestCase):
    def test_partial_round_is_in_progress_and_upcoming_returns_remaining_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_dir = Path(tmpdir)
            round_0 = pd.DataFrame(
                [
                    {"team": "Team A", "opponent": "Team B", "venue": "V1", "date": "2026-03-01", "is_home": 1},
                    {"team": "Team B", "opponent": "Team A", "venue": "V1", "date": "2026-03-01", "is_home": 0},
                    {"team": "Team C", "opponent": "Team D", "venue": "V2", "date": "2026-03-02", "is_home": 1},
                    {"team": "Team D", "opponent": "Team C", "venue": "V2", "date": "2026-03-02", "is_home": 0},
                ]
            )
            round_1 = pd.DataFrame(
                [
                    {"team": "Team E", "opponent": "Team F", "venue": "V3", "date": "2026-03-08", "is_home": 1},
                    {"team": "Team F", "opponent": "Team E", "venue": "V3", "date": "2026-03-08", "is_home": 0},
                ]
            )
            round_0.to_csv(fixture_dir / "round_0_2026.csv", index=False)
            round_1.to_csv(fixture_dir / "round_1_2026.csv", index=False)

            matches = pd.DataFrame(
                [
                    {
                        "year": 2026,
                        "round_number": 0,
                        "match_id": 100,
                        "home_team": "Team A",
                        "away_team": "Team B",
                        "venue": "V1",
                        "date": "2026-03-01",
                        "home_score": 90,
                        "away_score": 70,
                    }
                ]
            )
            fake_cache = SimpleNamespace(
                player_games=pd.DataFrame(columns=["year", "round_number"]),
                matches=matches,
                sequential_store=None,
                store=None,
            )

            with patch("api.services.season_service.DataCache.get", return_value=fake_cache), \
                 patch("config.FIXTURES_DIR", fixture_dir), \
                 patch("config.DATA_DIR", fixture_dir):
                upcoming = season_service.get_upcoming_matches(2026)
                schedule = season_service.get_season_schedule(2026)
                summary = season_service.get_season_summary(2026)

        self.assertEqual(upcoming["round_number"], 0)
        self.assertEqual(len(upcoming["matches"]), 1)
        self.assertEqual(upcoming["matches"][0]["home_team"], "Team C")

        rounds = {row["round_number"]: row["status"] for row in schedule["rounds"]}
        self.assertEqual(rounds[0], "in_progress")
        self.assertEqual(rounds[1], "upcoming")
        self.assertEqual(summary["completed_rounds"], 0)
        self.assertEqual(summary["current_round"], 0)

    def test_predictions_history_uses_match_id_when_available(self):
        fake_cache = SimpleNamespace(
            matches=pd.DataFrame(
                [
                    {
                        "year": 2025,
                        "round_number": 1,
                        "match_id": 10,
                        "home_team": "Team A",
                        "away_team": "Team B",
                        "home_score": 90,
                        "away_score": 70,
                    }
                ]
            ),
            sequential_store=_FakeStore(
                predictions={
                    (2025, 1): pd.DataFrame(
                        [
                            {
                                "player": "Player, One",
                                "team": "Team A",
                                "match_id": 10,
                                "opponent": "Team B",
                                "venue": "V1",
                                "predicted_goals": 1.2,
                            },
                            {
                                "player": "Player, One",
                                "team": "Team A",
                                "match_id": 11,
                                "opponent": "Team C",
                                "venue": "V2",
                                "predicted_goals": 2.4,
                            },
                        ]
                    )
                },
                outcomes={
                    (2025, 1): pd.DataFrame(
                        [
                            {
                                "player": "Player, One",
                                "team": "Team A",
                                "match_id": 10,
                                "actual_goals": 1,
                            },
                            {
                                "player": "Player, One",
                                "team": "Team A",
                                "match_id": 11,
                                "actual_goals": 3,
                            },
                        ]
                    )
                },
            ),
            store=None,
        )

        with patch("api.services.season_service.DataCache.get", return_value=fake_cache), \
             patch("api.services.season_service._build_round_progress", return_value={1: {"played_matches": 1, "fixture_matches": 1}}):
            history = season_service.get_predictions_history(2025)

        self.assertEqual(len(history["entries"]), 2)
        self.assertEqual(history["summary"]["total_predictions"], 2)
        self.assertEqual(sorted(entry["match_id"] for entry in history["entries"]), [10, 11])

    def test_season_schedule_falls_back_to_matches_when_fixtures_missing(self):
        fake_cache = SimpleNamespace(
            matches=pd.DataFrame(
                [
                    {
                        "year": 2025,
                        "round_number": 1,
                        "match_id": 10,
                        "home_team": "Team A",
                        "away_team": "Team B",
                        "venue": "V1",
                        "date": "2025-03-20T19:20:00",
                        "home_score": 90,
                        "away_score": 70,
                    },
                    {
                        "year": 2025,
                        "round_number": 1,
                        "match_id": 11,
                        "home_team": "Team C",
                        "away_team": "Team D",
                        "venue": "V2",
                        "date": "2025-03-21T19:50:00",
                        "home_score": 81,
                        "away_score": 80,
                    },
                ]
            ),
            sequential_store=None,
            store=None,
        )

        with patch("api.services.season_service.DataCache.get", return_value=fake_cache), \
             patch("api.services.season_service._get_fixture_round_files", return_value=[]):
            schedule = season_service.get_season_schedule(2025)

        self.assertEqual(len(schedule["rounds"]), 1)
        self.assertEqual(schedule["rounds"][0]["round_number"], 1)
        self.assertEqual(schedule["rounds"][0]["status"], "completed")
        self.assertEqual(len(schedule["rounds"][0]["matches"]), 2)
        self.assertEqual(schedule["rounds"][0]["matches"][0]["match_id"], 10)
        self.assertEqual(schedule["rounds"][0]["matches"][1]["match_id"], 11)


class TestReportingLogic(unittest.TestCase):
    def test_merge_pred_outcome_falls_back_to_round_aware_keys(self):
        predictions = pd.DataFrame(
            [
                {
                    "player": "Player, One",
                    "team": "Team A",
                    "round_number": 0,
                    "match_id": -1,
                    "predicted_goals": 1.0,
                },
                {
                    "player": "Player, One",
                    "team": "Team A",
                    "round_number": 1,
                    "match_id": -2,
                    "predicted_goals": 2.0,
                },
            ]
        )
        outcomes = pd.DataFrame(
            [
                {
                    "player": "Player, One",
                    "team": "Team A",
                    "round_number": 0,
                    "match_id": 100,
                    "actual_goals": 1,
                },
                {
                    "player": "Player, One",
                    "team": "Team A",
                    "round_number": 1,
                    "match_id": 101,
                    "actual_goals": 3,
                },
            ]
        )

        merged = analysis._merge_pred_outcome(predictions, outcomes)

        self.assertEqual(len(merged), 2)
        self.assertEqual(sorted(merged["round_number"].tolist()), [0, 1])
        actual_by_round = dict(zip(merged["round_number"], merged["actual_goals"]))
        self.assertEqual(actual_by_round[0], 1)
        self.assertEqual(actual_by_round[1], 3)

    def test_season_report_and_hit_rates_support_round_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LearningStore(base_dir=tmpdir)

            predictions = pd.DataFrame(
                [
                    {
                        "player": f"Player {i}",
                        "team": "Team A",
                        "match_id": -1,
                        "predicted_goals": 1.0 if i < 5 else 0.0,
                        "p_scorer": 0.8 if i < 5 else 0.2,
                    }
                    for i in range(10)
                ]
            )
            outcomes = pd.DataFrame(
                [
                    {
                        "player": f"Player {i}",
                        "team": "Team A",
                        "match_id": 100,
                        "actual_goals": 1 if i < 5 else 0,
                    }
                    for i in range(10)
                ]
            )

            store.save_predictions(2026, 0, predictions)
            store.save_outcomes(2026, 0, outcomes)
            store.save_analysis(
                2026,
                0,
                {
                    "year": 2026,
                    "round": 0,
                    "summary": {
                        "goals_mae": 0.0,
                        "scorer_auc": 1.0,
                        "calibration_ratio": 1.0,
                        "n_players": 10,
                        "threshold_metrics": {},
                    },
                    "model_improvement": {},
                    "weather_impact": {},
                    "game_results": [],
                    "streaks": {
                        "new_hot": [],
                        "continued_hot": [],
                        "broken_hot": [],
                        "new_cold": [],
                    },
                },
            )

            report = analysis.generate_season_report(store, 2026)
            hit_rates = pipeline._compute_hit_rates(store, 2026)

        self.assertEqual(report["rounds_analyzed"], 1)
        self.assertIn("1plus_goals", report["threshold_evaluation"])
        self.assertEqual(report["threshold_evaluation"]["1plus_goals"]["n"], 10)
        self.assertIn("1plus_goals", hit_rates)
        self.assertEqual(hit_rates["1plus_goals"]["n"], 10)

    def test_streak_summary_uses_previous_available_round_after_round_zero(self):
        store = _FakeStore(
            outcomes={
                (2026, 0): pd.DataFrame(
                    [
                        {"player": "Player, One", "team": "Team A", "actual_goals": 1},
                    ]
                ),
                (2026, 1): pd.DataFrame(
                    [
                        {"player": "Player, One", "team": "Team A", "actual_goals": 1},
                    ]
                ),
            },
            rounds=[(2026, 0), (2026, 1)],
        )

        summary = analysis._streak_summary(store, 2026, 1)

        self.assertNotIn("note", summary)
        self.assertEqual(summary["new_hot"], [])


if __name__ == "__main__":
    unittest.main()
