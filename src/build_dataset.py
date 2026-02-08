"""Build match-level dataset from team-level statistics."""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import PROCESSED_DATA_DIR

def clean_possession(value):
    """Convert possession from string percentage to float."""
    if pd.isna(value) or value == '':
        return 0.0
    if isinstance(value, str):
        # Remove % sign if present
        value = value.replace('%', '').strip()
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def clean_numeric(value):
    """Convert value to numeric, return 0 if missing or invalid."""
    if pd.isna(value) or value == '':
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

def main():
    # Read team-level stats
    input_file = f"{PROCESSED_DATA_DIR}/team_match_stats.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        print("Please run fetch_api.py first")
        sys.exit(1)

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} team-match records")

    # Separate home and away teams
    home_df = df[df['home_away'] == 'home'].copy()
    away_df = df[df['home_away'] == 'away'].copy()

    print(f"Home records: {len(home_df)}")
    print(f"Away records: {len(away_df)}")

    # Rename columns for home team
    home_df = home_df.rename(columns={
        'team_id': 'home_team_id',
        'team_name': 'home_team_name',
        'goals': 'home_goals',
        'shots_total': 'home_shots_total',
        'shots_on_target': 'home_shots_on_target',
        'possession': 'home_possession',
        'passes_total': 'home_passes_total',
        'passes_accurate': 'home_passes_accurate',
        'fouls': 'home_fouls',
        'yellow_cards': 'home_yellow_cards',
        'red_cards': 'home_red_cards',
        'offsides': 'home_offsides',
        'corners': 'home_corners',
    })

    # Rename columns for away team
    away_df = away_df.rename(columns={
        'team_id': 'away_team_id',
        'team_name': 'away_team_name',
        'goals': 'away_goals',
        'shots_total': 'away_shots_total',
        'shots_on_target': 'away_shots_on_target',
        'possession': 'away_possession',
        'passes_total': 'away_passes_total',
        'passes_accurate': 'away_passes_accurate',
        'fouls': 'away_fouls',
        'yellow_cards': 'away_yellow_cards',
        'red_cards': 'away_red_cards',
        'offsides': 'away_offsides',
        'corners': 'away_corners',
    })

    # Keep only relevant columns
    home_cols = ['fixture_id', 'date', 'league', 'season',
                 'home_team_id', 'home_team_name', 'home_goals',
                 'home_shots_total', 'home_shots_on_target', 'home_possession',
                 'home_passes_total', 'home_passes_accurate', 'home_fouls',
                 'home_yellow_cards', 'home_red_cards', 'home_offsides', 'home_corners']

    away_cols = ['fixture_id',
                 'away_team_id', 'away_team_name', 'away_goals',
                 'away_shots_total', 'away_shots_on_target', 'away_possession',
                 'away_passes_total', 'away_passes_accurate', 'away_fouls',
                 'away_yellow_cards', 'away_red_cards', 'away_offsides', 'away_corners']

    home_df = home_df[home_cols]
    away_df = away_df[away_cols]

    # Merge on fixture_id
    print("\nMerging home and away data...")
    match_df = pd.merge(home_df, away_df, on='fixture_id', how='inner')
    print(f"Created {len(match_df)} match records")

    # Clean numeric columns
    print("\nCleaning data...")

    # Convert possession from percentage strings to floats
    match_df['home_possession'] = match_df['home_possession'].apply(clean_possession)
    match_df['away_possession'] = match_df['away_possession'].apply(clean_possession)

    # Convert other numeric columns
    numeric_cols = [
        'home_goals', 'home_shots_total', 'home_shots_on_target',
        'home_passes_total', 'home_passes_accurate', 'home_fouls',
        'home_yellow_cards', 'home_red_cards', 'home_offsides', 'home_corners',
        'away_goals', 'away_shots_total', 'away_shots_on_target',
        'away_passes_total', 'away_passes_accurate', 'away_fouls',
        'away_yellow_cards', 'away_red_cards', 'away_offsides', 'away_corners'
    ]

    for col in numeric_cols:
        match_df[col] = match_df[col].apply(clean_numeric)

    # Save match dataset
    output_file = f"{PROCESSED_DATA_DIR}/match_dataset.csv"
    match_df.to_csv(output_file, index=False)
    print(f"\nSaved match dataset to {output_file}")
    print(f"Dataset shape: {match_df.shape}")
    print(f"\nColumns: {list(match_df.columns)}")

    # Show sample statistics
    print(f"\nSample statistics:")
    print(f"  Average home goals: {match_df['home_goals'].mean():.2f}")
    print(f"  Average away goals: {match_df['away_goals'].mean():.2f}")
    print(f"  Average home possession: {match_df['home_possession'].mean():.1f}%")
    print(f"  Average away possession: {match_df['away_possession'].mean():.1f}%")
    print(f"  Average home shots: {match_df['home_shots_total'].mean():.1f}")
    print(f"  Average away shots: {match_df['away_shots_total'].mean():.1f}")

    # Check for missing values
    missing = match_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")

if __name__ == '__main__':
    main()
