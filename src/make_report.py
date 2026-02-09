"""Generate visualizations and markdown report."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, LEAGUE_ID, SEASON

def create_actual_vs_predicted_plot(df, output_path):
    """Create scatter plots of actual vs predicted goals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Home goals
    ax1.scatter(df['predicted_home_goals'], df['actual_home_goals'], alpha=0.5, s=50)
    ax1.plot([0, 6], [0, 6], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Predicted Home Goals', fontsize=11)
    ax1.set_ylabel('Actual Home Goals', fontsize=11)
    ax1.set_title('Home Goals: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Calculate R²
    home_r2 = r2_score(df['actual_home_goals'], df['predicted_home_goals'])
    ax1.text(0.05, 0.95, f'R² = {home_r2:.3f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Away goals
    ax2.scatter(df['predicted_away_goals'], df['actual_away_goals'], alpha=0.5, s=50, color='orange')
    ax2.plot([0, 6], [0, 6], 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('Predicted Away Goals', fontsize=11)
    ax2.set_ylabel('Actual Away Goals', fontsize=11)
    ax2.set_title('Away Goals: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Calculate R²
    away_r2 = r2_score(df['actual_away_goals'], df['predicted_away_goals'])
    ax2.text(0.05, 0.95, f'R² = {away_r2:.3f}', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_goal_diff_error_plot(df, output_path):
    """Create histogram of goal difference errors."""
    # Recalculate with sign (not absolute)
    goal_diff_error = (df['fair_goal_diff'] - df['actual_goal_diff'])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(goal_diff_error, bins=range(-6, 7), edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Goal Difference Error (Fair - Actual)', fontsize=11)
    ax.set_ylabel('Number of Matches', fontsize=11)
    ax.set_title('Distribution of Goal Difference Errors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add stats text
    mean_error = goal_diff_error.mean()
    std_error = goal_diff_error.std()
    ax.text(0.02, 0.98, f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_team_luck_plot(team_luck_df, output_path):
    """Create bar chart of team luck."""
    # Sort by net lucky results and take top 10 and bottom 10
    sorted_df = team_luck_df.sort_values('net_lucky_results', ascending=False)
    top_10 = sorted_df.head(10)
    bottom_10 = sorted_df.tail(10).sort_values('net_lucky_results')

    combined = pd.concat([top_10, bottom_10])

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['green' if x > 0 else 'red' for x in combined['net_lucky_results']]
    bars = ax.barh(range(len(combined)), combined['net_lucky_results'], color=colors, alpha=0.7)

    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined['team'], fontsize=9)
    ax.set_xlabel('Net Lucky Results (Wins/Draws)', fontsize=11)
    ax.set_title('Team Luck: Most Lucky vs Most Unlucky', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, combined['net_lucky_results'])):
        if val > 0:
            ax.text(val + 0.1, i, f'+{val}', va='center', fontsize=8)
        else:
            ax.text(val - 0.1, i, f'{val}', va='center', ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_markdown_report(df, team_luck_df, output_path):
    """Generate markdown report."""
    # Calculate metrics
    home_mae = mean_absolute_error(df['actual_home_goals'], df['predicted_home_goals'])
    home_rmse = np.sqrt(mean_squared_error(df['actual_home_goals'], df['predicted_home_goals']))
    home_r2 = r2_score(df['actual_home_goals'], df['predicted_home_goals'])

    away_mae = mean_absolute_error(df['actual_away_goals'], df['predicted_away_goals'])
    away_rmse = np.sqrt(mean_squared_error(df['actual_away_goals'], df['predicted_away_goals']))
    away_r2 = r2_score(df['actual_away_goals'], df['predicted_away_goals'])

    # Get top mismatches
    df['abs_goal_diff_error'] = df['goal_diff_error']
    top_mismatches = df.nlargest(20, 'abs_goal_diff_error')[
        ['home_team_name', 'away_team_name', 'actual_score', 'fair_score', 'goal_diff_error']
    ]

    # Get team luck rankings
    luckiest = team_luck_df.nlargest(10, 'net_lucky_results')
    unluckiest = team_luck_df.nsmallest(10, 'net_lucky_results')

    # Write markdown
    with open(output_path, 'w') as f:
        f.write(f"# Fair-Game Analysis Report\n\n")
        f.write(f"**League:** {LEAGUE_ID} (Premier League)  \n")
        f.write(f"**Season:** {SEASON}  \n")
        f.write(f"**Matches Analyzed:** {len(df)}  \n\n")

        f.write(f"---\n\n")

        f.write(f"## Overview\n\n")
        f.write(f"This report analyzes {len(df)} Premier League matches from the {SEASON} season. ")
        f.write(f"We trained simple linear regression models to predict 'fair' scorelines based on ")
        f.write(f"match statistics (shots, shots on target, possession) and compared them to actual results.\n\n")

        f.write(f"## Model Performance\n\n")
        f.write(f"### Home Goals Model\n")
        f.write(f"- **MAE:** {home_mae:.3f}\n")
        f.write(f"- **RMSE:** {home_rmse:.3f}\n")
        f.write(f"- **R²:** {home_r2:.3f}\n\n")

        f.write(f"### Away Goals Model\n")
        f.write(f"- **MAE:** {away_mae:.3f}\n")
        f.write(f"- **RMSE:** {away_rmse:.3f}\n")
        f.write(f"- **R²:** {away_r2:.3f}\n\n")

        f.write(f"## Visualizations\n\n")

        f.write(f"### Actual vs Predicted Goals\n\n")
        f.write(f"![Actual vs Predicted](figures/actual_vs_predicted.png)\n\n")

        f.write(f"### Goal Difference Error Distribution\n\n")
        f.write(f"![Goal Diff Error](figures/goal_diff_error_hist.png)\n\n")

        f.write(f"### Team Luck Rankings\n\n")
        f.write(f"![Team Luck](figures/team_luck.png)\n\n")

        f.write(f"## Top 20 Biggest Mismatches\n\n")
        f.write(f"Matches where the fair scoreline differed most from the actual result:\n\n")
        f.write(f"| Home | Away | Actual | Fair | Error |\n")
        f.write(f"|------|------|--------|------|-------|\n")
        for _, row in top_mismatches.iterrows():
            f.write(f"| {row['home_team_name']} | {row['away_team_name']} | "
                   f"{row['actual_score']} | {row['fair_score']} | {row['goal_diff_error']:.0f} |\n")

        f.write(f"\n## Team Luck Analysis\n\n")
        f.write(f"### Luckiest Teams\n\n")
        f.write(f"Teams that got better results than their stats deserved:\n\n")
        f.write(f"| Team | Matches | Lucky Wins | Unlucky Losses | Net Lucky Results | Luck Rate |\n")
        f.write(f"|------|---------|------------|----------------|-------------------|----------|\n")
        for _, row in luckiest.iterrows():
            f.write(f"| {row['team']} | {row['matches']} | {row['lucky_wins']} | "
                   f"{row['unlucky_losses']} | {row['net_lucky_results']} | {row['luck_rate']:.1f}% |\n")

        f.write(f"\n### Unluckiest Teams\n\n")
        f.write(f"Teams that got worse results than their stats deserved:\n\n")
        f.write(f"| Team | Matches | Lucky Wins | Unlucky Losses | Net Lucky Results | Luck Rate |\n")
        f.write(f"|------|---------|------------|----------------|-------------------|----------|\n")
        for _, row in unluckiest.iterrows():
            f.write(f"| {row['team']} | {row['matches']} | {row['lucky_wins']} | "
                   f"{row['unlucky_losses']} | {row['net_lucky_results']} | {row['luck_rate']:.1f}% |\n")

        f.write(f"\n---\n\n")
        f.write(f"*Generated by fair-game pipeline*\n")

    print(f"Saved: {output_path}")

def main():
    # Read data
    fair_scores_file = f"{PROCESSED_DATA_DIR}/fair_scores.csv"
    team_luck_file = f"{PROCESSED_DATA_DIR}/team_luck.csv"

    if not os.path.exists(fair_scores_file):
        print(f"ERROR: {fair_scores_file} not found")
        print("Please run predict_fair_score.py first")
        sys.exit(1)

    if not os.path.exists(team_luck_file):
        print(f"ERROR: {team_luck_file} not found")
        print("Please run predict_fair_score.py first")
        sys.exit(1)

    print("Reading data...")
    df = pd.read_csv(fair_scores_file)
    team_luck_df = pd.read_csv(team_luck_file)
    print(f"Loaded {len(df)} matches")
    print(f"Loaded {len(team_luck_df)} teams")

    # Create output directories
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\nGenerating visualizations...")
    create_actual_vs_predicted_plot(df, f"{FIGURES_DIR}/actual_vs_predicted.png")
    create_goal_diff_error_plot(df, f"{FIGURES_DIR}/goal_diff_error_hist.png")
    create_team_luck_plot(team_luck_df, f"{FIGURES_DIR}/team_luck.png")

    # Generate report
    print("\nGenerating markdown report...")
    report_path = f"{REPORTS_DIR}/report.md"
    generate_markdown_report(df, team_luck_df, report_path)

    print(f"\n=== Report Complete ===")
    print(f"View the report at: {report_path}")
    print(f"Figures saved to: {FIGURES_DIR}/")

if __name__ == '__main__':
    main()
