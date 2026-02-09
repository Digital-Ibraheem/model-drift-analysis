"""Generate predictions and compute fair scorelines."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import PROCESSED_DATA_DIR, MODELS_DIR


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate fair score predictions'
    )
    parser.add_argument(
        '--model', type=str, choices=['linear', 'poisson'],
        default='linear', help='Model type to use (default: linear)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine model files based on type
    suffix = f"_{args.model}" if args.model != 'linear' else ""
    home_model_file = f"{MODELS_DIR}/home_model{suffix}.pkl"
    away_model_file = f"{MODELS_DIR}/away_model{suffix}.pkl"
    metadata_file = f"{MODELS_DIR}/model_metadata{suffix}.pkl"

    # Check model files exist
    for f in [home_model_file, away_model_file]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found")
            print("Please run train_model.py first")
            sys.exit(1)

    print(f"Loading models (type: {args.model})...")
    home_model = joblib.load(home_model_file)
    away_model = joblib.load(away_model_file)
    print(f"Loaded: {home_model_file}")
    print(f"Loaded: {away_model_file}")

    # Load metadata if available, otherwise use defaults
    if os.path.exists(metadata_file):
        metadata = joblib.load(metadata_file)
        home_features = metadata['home_features']
        away_features = metadata['away_features']
        print(f"Loaded metadata: {metadata_file}")
        print(f"Feature set: {metadata.get('feature_set', 'unknown')}")
    else:
        # Fallback to legacy features
        print("No metadata found, using default features")
        home_features = ['home_shots_total', 'home_shots_on_target', 'home_possession', 'home_indicator']
        away_features = ['away_shots_total', 'away_shots_on_target', 'away_possession']

    # Read match dataset
    input_file = f"{PROCESSED_DATA_DIR}/match_dataset.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        print("Please run build_dataset.py first")
        sys.exit(1)

    print(f"\nReading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} matches")

    # Prepare features for home model
    if 'home_indicator' in home_features and 'home_indicator' not in df.columns:
        df['home_indicator'] = 1

    # Check which features are available
    available_home = [f for f in home_features if f in df.columns]
    available_away = [f for f in away_features if f in df.columns]

    missing_home = [f for f in home_features if f not in df.columns]
    missing_away = [f for f in away_features if f not in df.columns]

    if missing_home:
        print(f"Warning: Missing home features: {missing_home}")
    if missing_away:
        print(f"Warning: Missing away features: {missing_away}")

    X_home = df[available_home].fillna(0)
    X_away = df[available_away].fillna(0)

    # Generate predictions
    print("\nGenerating predictions...")
    predicted_home_goals = home_model.predict(X_home)
    predicted_away_goals = away_model.predict(X_away)

    # Compute fair scorelines (round and clip to [0, 6])
    print("Computing fair scorelines...")
    fair_home_goals = np.clip(np.round(predicted_home_goals), 0, 6).astype(int)
    fair_away_goals = np.clip(np.round(predicted_away_goals), 0, 6).astype(int)

    # Create output dataframe
    output_df = pd.DataFrame({
        'fixture_id': df['fixture_id'],
        'date': df['date'],
        'home_team_name': df['home_team_name'],
        'away_team_name': df['away_team_name'],
        'actual_home_goals': df['home_goals'],
        'actual_away_goals': df['away_goals'],
        'predicted_home_goals': predicted_home_goals,
        'predicted_away_goals': predicted_away_goals,
        'fair_home_goals': fair_home_goals,
        'fair_away_goals': fair_away_goals,
    })

    # Add some derived columns for analysis
    output_df['actual_score'] = output_df['actual_home_goals'].astype(str) + '-' + output_df['actual_away_goals'].astype(str)
    output_df['fair_score'] = output_df['fair_home_goals'].astype(str) + '-' + output_df['fair_away_goals'].astype(str)
    output_df['actual_goal_diff'] = output_df['actual_home_goals'] - output_df['actual_away_goals']
    output_df['fair_goal_diff'] = output_df['fair_home_goals'] - output_df['fair_away_goals']
    output_df['goal_diff_error'] = abs(output_df['fair_goal_diff'] - output_df['actual_goal_diff'])

    # Save to CSV
    output_file = f"{PROCESSED_DATA_DIR}/fair_scores.csv"
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")
    print(f"Dataset shape: {output_df.shape}")

    # Show summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Matches processed: {len(output_df)}")
    print(f"\nActual goals:")
    print(f"  Home: {output_df['actual_home_goals'].mean():.2f} ± {output_df['actual_home_goals'].std():.2f}")
    print(f"  Away: {output_df['actual_away_goals'].mean():.2f} ± {output_df['actual_away_goals'].std():.2f}")
    print(f"\nPredicted goals (before rounding):")
    print(f"  Home: {output_df['predicted_home_goals'].mean():.2f} ± {output_df['predicted_home_goals'].std():.2f}")
    print(f"  Away: {output_df['predicted_away_goals'].mean():.2f} ± {output_df['predicted_away_goals'].std():.2f}")
    print(f"\nFair scoreline (after rounding/clipping):")
    print(f"  Home: {output_df['fair_home_goals'].mean():.2f} ± {output_df['fair_home_goals'].std():.2f}")
    print(f"  Away: {output_df['fair_away_goals'].mean():.2f} ± {output_df['fair_away_goals'].std():.2f}")

    # Show some example predictions
    print(f"\n=== Sample Predictions ===")
    sample = output_df[['home_team_name', 'away_team_name', 'actual_score', 'fair_score', 'goal_diff_error']].head(10)
    print(sample.to_string(index=False))

    # Show biggest mismatches
    print(f"\n=== Biggest Mismatches (by goal difference error) ===")
    output_df['abs_goal_diff_error'] = output_df['goal_diff_error'].abs()
    mismatches = output_df.nlargest(5, 'abs_goal_diff_error')[
        ['home_team_name', 'away_team_name', 'actual_score', 'fair_score', 'goal_diff_error']
    ]
    print(mismatches.to_string(index=False))

    # Calculate team-level luck based on result flips
    print(f"\n=== Team Luck Analysis ===")

    def classify_result(goal_diff):
        """Classify result as W/D/L based on goal difference."""
        if goal_diff > 0:
            return 'W'
        elif goal_diff < 0:
            return 'L'
        else:
            return 'D'

    # Create rows for each team's home matches
    home_luck = output_df[['home_team_name', 'actual_goal_diff', 'fair_goal_diff']].copy()
    home_luck.columns = ['team', 'actual_diff', 'fair_diff']

    # Create rows for each team's away matches (flip the sign)
    away_luck = output_df[['away_team_name', 'actual_goal_diff', 'fair_goal_diff']].copy()
    away_luck.columns = ['team', 'actual_diff', 'fair_diff']
    away_luck['actual_diff'] = -away_luck['actual_diff']
    away_luck['fair_diff'] = -away_luck['fair_diff']

    # Combine home and away
    team_luck = pd.concat([home_luck, away_luck], ignore_index=True)

    # Classify results
    team_luck['actual_result'] = team_luck['actual_diff'].apply(classify_result)
    team_luck['fair_result'] = team_luck['fair_diff'].apply(classify_result)

    # Calculate luck metrics
    team_luck['lucky_win'] = ((team_luck['actual_result'] == 'W') &
                              (team_luck['fair_result'].isin(['D', 'L']))).astype(int)
    team_luck['unlucky_loss'] = ((team_luck['actual_result'] == 'L') &
                                 (team_luck['fair_result'].isin(['D', 'W']))).astype(int)
    team_luck['lucky_draw'] = ((team_luck['actual_result'] == 'D') &
                               (team_luck['fair_result'] == 'L')).astype(int)
    team_luck['unlucky_draw'] = ((team_luck['actual_result'] == 'D') &
                                 (team_luck['fair_result'] == 'W')).astype(int)

    # Aggregate by team
    team_stats = team_luck.groupby('team').agg({
        'actual_result': 'count',  # total matches
        'lucky_win': 'sum',
        'unlucky_loss': 'sum',
        'lucky_draw': 'sum',
        'unlucky_draw': 'sum'
    }).reset_index()

    team_stats.columns = ['team', 'matches', 'lucky_wins', 'unlucky_losses', 'lucky_draws', 'unlucky_draws']

    # Calculate net luck and luck rate
    team_stats['net_lucky_results'] = (team_stats['lucky_wins'] + team_stats['lucky_draws'] -
                                       team_stats['unlucky_losses'] - team_stats['unlucky_draws'])
    team_stats['luck_rate'] = (team_stats['net_lucky_results'] / team_stats['matches'] * 100).round(1)

    # Sort by net lucky results
    team_stats = team_stats.sort_values('net_lucky_results', ascending=False)

    # Show luckiest teams
    print(f"\nLuckiest Teams (won matches they should have lost/drawn):")
    luckiest = team_stats.head(5)[['team', 'matches', 'lucky_wins', 'unlucky_losses', 'net_lucky_results', 'luck_rate']]
    print(luckiest.to_string(index=False))

    # Show unluckiest teams
    print(f"\nUnluckiest Teams (lost matches they should have won/drawn):")
    unluckiest = team_stats.tail(5).sort_values('net_lucky_results')[['team', 'matches', 'lucky_wins', 'unlucky_losses', 'net_lucky_results', 'luck_rate']]
    print(unluckiest.to_string(index=False))

    # Save team luck analysis
    team_luck_file = f"{PROCESSED_DATA_DIR}/team_luck.csv"
    team_stats.to_csv(team_luck_file, index=False)
    print(f"\nSaved team luck analysis to {team_luck_file}")

if __name__ == '__main__':
    main()
