# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fair-game is a soccer analytics project that fetches match statistics from the API-Football API, trains simple linear regression models to predict "fair" scorelines based on match stats (shots, possession, etc.), and compares predictions to actual results.

## Architecture

The project follows a **linear pipeline architecture** with 6 sequential steps, each implemented as a standalone script:

1. **smoke_test.py** → Verifies API credentials
2. **fetch_api.py** → Downloads fixtures and team statistics from API, saves raw JSON and produces `team_match_stats.csv`
3. **build_dataset.py** → Pivots team-level stats into match-level format for training
4. **train_model.py** → Trains two separate LinearRegression models (home goals, away goals)
5. **predict_fair_score.py** → Generates predictions and computes fair scorelines (rounded, clipped [0,6])
6. **make_report.py** → Creates visualizations and markdown report

**Key Design Principle**: Single path, minimal moving parts, no branching. Each script is independent and runnable via `python src/<script>.py`.

### Data Flow

```
API-Football API
    ↓
data/raw/*.json (raw API responses)
    ↓
data/processed/team_match_stats.csv (one row per team per match)
    ↓
data/processed/match_dataset.csv (one row per match, home/away side-by-side)
    ↓
models/home_model.pkl, models/away_model.pkl
    ↓
data/processed/fair_scores.csv
    ↓
reports/report.md + reports/figures/*.png
```

### Configuration

All constants live in `fair_game/config.py`:
- `LEAGUE_ID` and `SEASON` define what data to fetch (currently Premier League 2023)
- File path constants define where data/models/reports are stored
- To analyze a different league/season, modify these constants and re-run the pipeline

## Running the Pipeline

The pipeline must be executed in order. Each script depends on outputs from previous steps.

```bash
# 0. Setup (first time only)
pip install -r requirements.txt
# Add your API key to .env: FOOTBALL_API_KEY=your_key_here

# 1. Verify API credentials
python src/smoke_test.py

# 2. Fetch data from API (~5-10 min depending on number of fixtures)
python src/fetch_api.py

# 3. Transform to match-level dataset
python src/build_dataset.py

# 4. Train models
python src/train_model.py

# 5. Generate predictions
python src/predict_fair_score.py

# 6. Create report
python src/make_report.py

# 7. View results
open reports/report.md
```

## API Integration

The project uses API-Football (API-Sports) v3:
- **Base URL**: `https://v3.football.api-sports.io`
- **Authentication**: Header `x-apisports-key` with API key from `.env`
- **Key endpoints**:
  - `GET /status` - Verify credentials, check rate limits
  - `GET /fixtures?league={id}&season={year}` - Fetch all fixtures
  - `GET /fixtures/statistics?fixture={id}` - Fetch per-team statistics

Rate limiting: 1-second delay between statistics requests to respect API limits.

## Code Organization

- **src/** - Executable scripts (6 pipeline steps). Each script is self-contained and can be run independently after dependencies are satisfied.
- **fair_game/** - Python package for shared configuration. Currently only contains `config.py`.
- **data/** - Git-ignored. Contains raw API responses, processed CSVs, organized by pipeline stage.
- **models/** - Git-ignored. Contains trained scikit-learn models serialized with joblib.
- **reports/** - Git-ignored figures/, but report.md can be committed if desired.

## Model Details

Two separate `sklearn.linear_model.LinearRegression` models:
- **Home model**: Predicts `home_goals` from home team stats (shots_total, shots_on_target, possession) + home_indicator constant
- **Away model**: Predicts `away_goals` from away team stats (shots_total, shots_on_target, possession)

No train/test split - trains on all available data. Missing values filled with 0.

Fair scoreline = round(prediction) then clip to [0, 6].

## Environment Setup

Required environment variables in `.env`:
```
FOOTBALL_API_KEY=your_api_key_here
FOOTBALL_API_HOST=v3.football.api-sports.io
```

The smoke test will fail with a clear error if the API key is not set or still has the placeholder value.
