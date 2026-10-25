"""Content-based music retrieval using TF-IDF and cosine similarity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "data" / "tracks.json"


@dataclass(frozen=True)
class Track:
    title: str
    artist: str
    genre: str
    mood: tuple[str, ...]
    activities: tuple[str, ...]
    description: str

    @classmethod
    def from_dict(cls, item: dict) -> "Track":
        return cls(
            title=str(item["title"]),
            artist=str(item["artist"]),
            genre=str(item["genre"]),
            mood=tuple(str(x) for x in item.get("mood", [])),
            activities=tuple(str(x) for x in item.get("activities", [])),
            description=str(item.get("description", "")),
        )

    def searchable_text(self) -> str:
        """Build the text representation used for retrieval."""
        return " ".join([
            self.title,
            self.artist,
            self.genre,
            " ".join(self.mood),
            " ".join(self.activities),
            self.description,
        ])


class MusicRecommender:
    """Rank catalog tracks by text similarity to a natural-language query."""

    def __init__(self, tracks: list[Track]):
        if not tracks:
            raise ValueError("The track catalog must not be empty.")
        self.tracks = tracks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.track_matrix = self.vectorizer.fit_transform(
            [track.searchable_text() for track in tracks]
        )

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CATALOG) -> "MusicRecommender":
        """Load tracks from a JSON catalog."""
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Catalog JSON must contain a list of tracks.")
        return cls([Track.from_dict(item) for item in data])

    def recommend(self, query: str, limit: int = 5) -> list[tuple[Track, float]]:
        """Return the top matching tracks and their cosine similarity scores."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        query_vector = self.vectorizer.transform([query.strip()])
        scores = cosine_similarity(query_vector, self.track_matrix).ravel()
        ranked_indices = sorted(
            range(len(self.tracks)),
            key=lambda index: (-float(scores[index]), self.tracks[index].title.lower()),
        )
        return [
            (self.tracks[index], float(scores[index]))
            for index in ranked_indices[: min(limit, len(self.tracks))]
        ]
