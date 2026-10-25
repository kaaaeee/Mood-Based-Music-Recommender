"""Command-line interface for the music recommender."""
from __future__ import annotations

import sys

from src.recommender import MusicRecommender


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Describe the music you want: ").strip()
    if not query:
        print("Please enter a non-empty query.")
        raise SystemExit(1)

    recommender = MusicRecommender.from_json()
    results = recommender.recommend(query, limit=5)
    print(f"\nQuery: {query}")
    print("\nRecommended tracks:")
    for position, (track, score) in enumerate(results, start=1):
        tags = ", ".join([track.genre, *track.mood[:2]])
        print(f"{position}. {track.title} — {track.artist}")
        print(f"   Similarity: {score:.3f} | {tags}")
        print(f"   Description: {track.description}")


if __name__ == "__main__":
    main()
