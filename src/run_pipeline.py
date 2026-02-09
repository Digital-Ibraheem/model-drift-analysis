"""Run the complete fair-game pipeline."""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import LEAGUE_ID, SEASON

SCRIPT_DIR = Path(__file__).parent

PIPELINE_STEPS = [
    ("smoke_test.py", "Verifying API credentials"),
    ("fetch_api.py", "Fetching match data from API"),
    ("build_dataset.py", "Building match-level dataset"),
    ("train_model.py", "Training prediction models"),
    ("predict_fair_score.py", "Generating predictions"),
    ("make_report.py", "Creating report and visualizations"),
]


def run_step(script_name, description, extra_args=None, skip_on_error=False):
    """Run a pipeline step and return success status.

    Args:
        script_name: Name of the script to run
        description: Human-readable description of the step
        extra_args: Additional CLI arguments to pass
        skip_on_error: If True, continue pipeline on failure

    Returns:
        bool: True if step succeeded
    """
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Script: {script_name}")
    print('='*60 + "\n")

    cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {script_name} failed with exit code {e.returncode}")
        if not skip_on_error:
            print("Pipeline halted. Fix the error and re-run.")
            sys.exit(1)
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the fair-game pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python src/run_pipeline.py

  # Skip API fetch (use existing data)
  python src/run_pipeline.py --skip-fetch

  # Use Poisson model with extended features
  python src/run_pipeline.py --skip-fetch --model poisson --features extended

  # Fetch data for La Liga 2024
  python src/run_pipeline.py --league 140 --season 2024
        """
    )
    parser.add_argument(
        '--skip-fetch', action='store_true',
        help='Skip API fetch (use existing data)'
    )
    parser.add_argument(
        '--skip-smoke', action='store_true',
        help='Skip smoke test'
    )
    parser.add_argument(
        '--model', type=str, choices=['linear', 'poisson'],
        default='linear', help='Model type to use (default: linear)'
    )
    parser.add_argument(
        '--features', type=str, choices=['basic', 'extended', 'xg'],
        default='basic', help='Feature set to use (default: basic)'
    )
    parser.add_argument(
        '--league', type=int, default=None,
        help=f'League ID to fetch (default: {LEAGUE_ID} from config)'
    )
    parser.add_argument(
        '--season', type=int, default=None,
        help=f'Season year to fetch (default: {SEASON} from config)'
    )
    parser.add_argument(
        '--no-split', action='store_true',
        help='Train on all data (no train/test split)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("FAIR-GAME PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Features: {args.features}")
    print(f"  League: {args.league or LEAGUE_ID}")
    print(f"  Season: {args.season or SEASON}")
    if args.skip_fetch:
        print("  [Skipping API fetch]")
    if args.skip_smoke:
        print("  [Skipping smoke test]")

    # Build extra args for various steps
    fetch_args = []
    if args.league:
        fetch_args.extend(['--league', str(args.league)])
    if args.season:
        fetch_args.extend(['--season', str(args.season)])

    train_args = ['--model', args.model, '--features', args.features]
    if args.no_split:
        train_args.append('--no-split')

    predict_args = ['--model', args.model]

    # Run pipeline
    for script, desc in PIPELINE_STEPS:
        # Skip steps if requested
        if script == "smoke_test.py" and args.skip_smoke:
            print(f"\n[SKIP] {script}")
            continue
        if script == "fetch_api.py" and args.skip_fetch:
            print(f"\n[SKIP] {script}")
            continue

        # Add extra args for specific steps
        extra = None
        if script == "fetch_api.py":
            extra = fetch_args if fetch_args else None
        elif script == "train_model.py":
            extra = train_args
        elif script == "predict_fair_score.py":
            extra = predict_args

        run_step(script, desc, extra_args=extra)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nView the report at: reports/report.md")


if __name__ == '__main__':
    main()
