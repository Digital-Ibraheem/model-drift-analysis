"""Train regression models to predict home and away goals."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_SETS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train goal prediction models'
    )
    parser.add_argument(
        '--model', type=str, choices=['linear', 'poisson'],
        default='linear', help='Model type (default: linear)'
    )
    parser.add_argument(
        '--features', type=str, choices=['basic', 'extended', 'xg'],
        default='basic', help='Feature set to use (default: basic)'
    )
    parser.add_argument(
        '--cv-folds', type=int, default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--no-split', action='store_true',
        help='Train on all data (legacy mode, no train/test split)'
    )
    return parser.parse_args()


def get_model(model_type):
    """Return appropriate sklearn model based on type."""
    if model_type == 'poisson':
        return PoissonRegressor(alpha=0.0, max_iter=1000)
    return LinearRegression()


def build_features(df, prefix, feature_set='basic'):
    """Build feature matrix for home or away team.

    Args:
        df: DataFrame with match data
        prefix: 'home' or 'away'
        feature_set: Name of feature set from config

    Returns:
        tuple: (X DataFrame, list of feature names used)
    """
    base_features = FEATURE_SETS.get(feature_set, FEATURE_SETS['basic'])

    # Map feature names to actual column names
    feature_cols = []
    for f in base_features:
        if f == 'expected_goals':
            col = f'{prefix}_xg'
        else:
            col = f'{prefix}_{f}'
        feature_cols.append(col)

    # Check which features exist in dataframe
    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]

    if missing:
        print(f"Warning: Missing features for {prefix}: {missing}")

    if not available:
        print(f"ERROR: No features available for {prefix}")
        sys.exit(1)

    X = df[available].fillna(0).copy()

    # Add home indicator for home model
    if prefix == 'home':
        X['home_indicator'] = 1

    return X, available


def evaluate_model(model, X, y, prefix=''):
    """Calculate and print evaluation metrics."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"{prefix}MAE:  {mae:.3f}")
    print(f"{prefix}RMSE: {rmse:.3f}")
    print(f"{prefix}R2:   {r2:.3f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}


def cross_validate_model(model_type, X, y, cv_folds, model_name):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Use MSE scoring for both model types for comparability
    model = get_model(model_type)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    print(f"\n{model_name} Cross-Validation ({cv_folds}-fold):")
    print(f"  RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std() * 2:.3f})")

    return rmse_scores


def print_coefficients(model, feature_names, model_name):
    """Print model coefficients for linear regression."""
    print(f"\n{model_name} Coefficients:")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"  {feature:25s}: {coef:8.4f}")
    print(f"  {'intercept':25s}: {model.intercept_:8.4f}")


def main():
    args = parse_args()

    # Load data
    input_file = f"{PROCESSED_DATA_DIR}/match_dataset.csv"
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        print("Please run build_dataset.py first")
        sys.exit(1)

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} matches")
    print(f"Model type: {args.model}")
    print(f"Feature set: {args.features}")

    # Build features
    X_home, home_features = build_features(df, 'home', args.features)
    X_away, away_features = build_features(df, 'away', args.features)
    y_home = df['home_goals']
    y_away = df['away_goals']

    print(f"\nHome features ({len(home_features)}): {home_features}")
    print(f"Away features ({len(away_features)}): {away_features}")
    print(f"Training samples: {len(X_home)}")

    # Cross-validation
    print(f"\n{'='*50}")
    print("Cross-Validation Results")
    print('='*50)
    cross_validate_model(args.model, X_home, y_home, args.cv_folds, "Home Goals")
    cross_validate_model(args.model, X_away, y_away, args.cv_folds, "Away Goals")

    # Train/test split (unless --no-split)
    if args.no_split:
        print(f"\n{'='*50}")
        print("Training on ALL data (legacy mode)")
        print('='*50)
        X_home_train, X_home_test = X_home, X_home
        X_away_train, X_away_test = X_away, X_away
        y_home_train, y_home_test = y_home, y_home
        y_away_train, y_away_test = y_away, y_away
    else:
        print(f"\n{'='*50}")
        print(f"Train/Test Split: {(1-args.test_size)*100:.0f}% / {args.test_size*100:.0f}%")
        print('='*50)
        X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(
            X_home, y_home, test_size=args.test_size, random_state=42
        )
        X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(
            X_away, y_away, test_size=args.test_size, random_state=42
        )
        print(f"Training samples: {len(X_home_train)}")
        print(f"Test samples: {len(X_home_test)}")

    # Train models
    print(f"\n{'='*50}")
    print("Training Models")
    print('='*50)

    home_model = get_model(args.model)
    home_model.fit(X_home_train, y_home_train)

    away_model = get_model(args.model)
    away_model.fit(X_away_train, y_away_train)

    # Evaluate on training set
    print("\n--- Home Model ---")
    print("[Train] ", end='')
    evaluate_model(home_model, X_home_train, y_home_train)

    print("\n--- Away Model ---")
    print("[Train] ", end='')
    evaluate_model(away_model, X_away_train, y_away_train)

    # Evaluate on test set (if split)
    if not args.no_split:
        print("\n--- Test Set Performance ---")
        print("\nHome Model [Test] ", end='')
        evaluate_model(home_model, X_home_test, y_home_test)
        print("\nAway Model [Test] ", end='')
        evaluate_model(away_model, X_away_test, y_away_test)

    # Print coefficients for linear model
    if args.model == 'linear':
        print_coefficients(home_model, list(X_home_train.columns), "Home Model")
        print_coefficients(away_model, list(X_away_train.columns), "Away Model")

    # Save models
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Save with model type suffix for non-default
    suffix = f"_{args.model}" if args.model != 'linear' else ""
    home_model_file = f"{MODELS_DIR}/home_model{suffix}.pkl"
    away_model_file = f"{MODELS_DIR}/away_model{suffix}.pkl"
    metadata_file = f"{MODELS_DIR}/model_metadata{suffix}.pkl"

    # Save metadata for prediction script
    metadata = {
        'model_type': args.model,
        'feature_set': args.features,
        'home_features': list(X_home_train.columns),
        'away_features': list(X_away_train.columns),
        'test_size': 0 if args.no_split else args.test_size,
        'cv_folds': args.cv_folds,
    }

    joblib.dump(home_model, home_model_file)
    joblib.dump(away_model, away_model_file)
    joblib.dump(metadata, metadata_file)

    print(f"\n{'='*50}")
    print("Models Saved")
    print('='*50)
    print(f"Home model: {home_model_file}")
    print(f"Away model: {away_model_file}")
    print(f"Metadata: {metadata_file}")


if __name__ == '__main__':
    main()
