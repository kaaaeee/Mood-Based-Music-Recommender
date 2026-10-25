# Mood-Based Music Recommender

A small NLP/search prototype that recommends tracks from a local catalog based on a natural-language request such as:

- `calm music for studying`
- `energetic songs for a workout`
- `music for a rainy night drive`
- `something dreamy and nostalgic`

The project ranks tracks by text similarity between the user's request and each track's description.

## What it demonstrates

- Text search with TF-IDF (`scikit-learn`)
- Cosine-similarity ranking
- Combining track title, artist, genre, mood, activity, and description into searchable text
- A small command-line interface
- Basic automated tests and a tiny offline evaluation set

## Important scope note

The included catalog is a **small hand-written demo catalog**, not a licensed or comprehensive music database. It contains illustrative track entries and is intended to demonstrate the retrieval approach. It does not play music or connect to a streaming service.

## Requirements

- Python 3.10+
- Internet is not required to run the project after installing dependencies.

## Run on Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.recommend
.venv\Scripts\python.exe -m src.recommend "calm music for studying"
.venv\Scripts\python.exe -m pytest
```

If the virtual environment is already active, you can also use `python -m src.recommend "music for a night drive"`.

## Example

```text
Query: calm music for studying

1. Weightless — Marconi Union
   Score: 0.31 | Ambient, calm, focus
   Why: Strong text match for calm, ambient, and study/focus terms.

2. Near Light — Ólafur Arnalds
   Score: 0.24 | Modern classical, reflective, focus
   Why: Matches reflective and focus-related language.
```

Scores depend on the catalog and vectorizer settings; they are similarity scores, not probabilities.

## Project structure

```text
mood-music-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── tracks.json
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   └── recommend.py
└── tests/
    └── test_recommender.py
```

## Next steps

Possible extensions: use sentence embeddings, build a conversational interface that keeps context between turns, evaluate against human relevance labels, or connect to a properly licensed music metadata API.
