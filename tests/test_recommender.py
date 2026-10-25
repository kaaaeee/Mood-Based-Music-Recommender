import pytest

from src.recommender import MusicRecommender


def make_recommender():
    return MusicRecommender.from_json()


def test_catalog_loads_and_has_tracks():
    recommender = make_recommender()
    assert len(recommender.tracks) >= 10


def test_workout_query_ranks_energetic_tracks():
    recommender = make_recommender()
    results = recommender.recommend("energetic music for a workout", limit=5)
    assert len(results) == 5
    top_titles = [track.title for track, _ in results[:3]]
    assert any(title in top_titles for title in ["Titanium", "Can't Hold Us", "Run Boy Run"])


def test_calm_study_query_returns_results():
    recommender = make_recommender()
    results = recommender.recommend("calm music for studying", limit=3)
    assert len(results) == 3
    assert all(0.0 <= score <= 1.0 for _, score in results)


def test_empty_query_is_rejected():
    recommender = make_recommender()
    with pytest.raises(ValueError):
        recommender.recommend("   ")


def test_limit_must_be_positive():
    recommender = make_recommender()
    with pytest.raises(ValueError):
        recommender.recommend("night drive", limit=0)
