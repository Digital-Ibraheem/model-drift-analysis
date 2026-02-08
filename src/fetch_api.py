"""Fetch fixtures and team statistics from API-Football."""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
import pandas as pd

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import (
    LEAGUE_ID, SEASON, BASE_URL,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)

def fetch_fixtures(api_key, api_host):
    """Fetch all fixtures for the configured league and season."""
    url = f"https://{api_host}/fixtures"
    headers = {'x-apisports-key': api_key}
    params = {
        'league': LEAGUE_ID,
        'season': SEASON
    }

    print(f"Fetching fixtures for league {LEAGUE_ID}, season {SEASON}...")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()

    if data.get('errors'):
        print(f"ERROR: {data['errors']}")
        sys.exit(1)

    fixtures = data.get('response', [])
    print(f"Found {len(fixtures)} fixtures")

    # Save raw response
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{RAW_DATA_DIR}/fixtures_{LEAGUE_ID}_{SEASON}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw fixtures to {output_file}")

    return fixtures

def fetch_fixture_statistics(fixture_id, api_key, api_host):
    """Fetch team statistics for a specific fixture."""
    url = f"https://{api_host}/fixtures/statistics"
    headers = {'x-apisports-key': api_key}
    params = {'fixture': fixture_id}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()

    if data.get('errors'):
        print(f"ERROR for fixture {fixture_id}: {data['errors']}")
        return None

    return data

def extract_stat_value(statistics, stat_type):
    """Extract a specific statistic value from the statistics array."""
    for stat in statistics:
        if stat.get('type') == stat_type:
            value = stat.get('value')
            # Handle percentage strings
            if isinstance(value, str) and value.endswith('%'):
                return value.replace('%', '')
            return value if value is not None else ''
    return ''

def process_fixture_stats(fixture_data, stats_data):
    """Process fixture and statistics data into team-level rows."""
    rows = []

    fixture_id = fixture_data['fixture']['id']
    date = fixture_data['fixture']['date']
    league = fixture_data['league']['name']
    season = fixture_data['league']['season']

    # Get teams
    home_team = fixture_data['teams']['home']
    away_team = fixture_data['teams']['away']

    # Get goals
    home_goals = fixture_data['goals']['home'] or 0
    away_goals = fixture_data['goals']['away'] or 0

    # Process statistics for each team
    if stats_data and stats_data.get('response'):
        for team_stats in stats_data['response']:
            team = team_stats['team']
            statistics = team_stats['statistics']

            # Determine home/away
            is_home = team['id'] == home_team['id']
            home_away = 'home' if is_home else 'away'
            goals = home_goals if is_home else away_goals

            # Extract key statistics
            row = {
                'fixture_id': fixture_id,
                'date': date,
                'league': league,
                'season': season,
                'team_id': team['id'],
                'team_name': team['name'],
                'home_away': home_away,
                'goals': goals,
                'shots_total': extract_stat_value(statistics, 'Total Shots'),
                'shots_on_target': extract_stat_value(statistics, 'Shots on Goal'),
                'possession': extract_stat_value(statistics, 'Ball Possession'),
                'passes_total': extract_stat_value(statistics, 'Total passes'),
                'passes_accurate': extract_stat_value(statistics, 'Passes accurate'),
                'fouls': extract_stat_value(statistics, 'Fouls'),
                'yellow_cards': extract_stat_value(statistics, 'Yellow Cards'),
                'red_cards': extract_stat_value(statistics, 'Red Cards'),
                'offsides': extract_stat_value(statistics, 'Offsides'),
                'corners': extract_stat_value(statistics, 'Corner Kicks'),
            }
            rows.append(row)

    return rows

def main():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('FOOTBALL_API_KEY')
    api_host = os.getenv('FOOTBALL_API_HOST', 'v3.football.api-sports.io')

    if not api_key or api_key == 'put_your_key_here':
        print("ERROR: FOOTBALL_API_KEY not set in .env file")
        sys.exit(1)

    # Fetch fixtures
    fixtures = fetch_fixtures(api_key, api_host)

    # Filter to only finished fixtures
    finished_fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']
    print(f"Processing {len(finished_fixtures)} finished fixtures")

    # Rate limit: 10 requests per minute
    BATCH_SIZE = 10
    WAIT_TIME = 60  # seconds

    num_batches = (len(finished_fixtures) + BATCH_SIZE - 1) // BATCH_SIZE
    estimated_minutes = num_batches
    print(f"Estimated time: ~{estimated_minutes} minutes ({BATCH_SIZE} requests/minute)")
    print(f"Processing in {num_batches} batches\n")

    # Fetch statistics for each fixture in batches
    all_rows = []
    batch_num = 0

    for batch_start in range(0, len(finished_fixtures), BATCH_SIZE):
        batch_num += 1
        batch_end = min(batch_start + BATCH_SIZE, len(finished_fixtures))
        batch = finished_fixtures[batch_start:batch_end]

        print(f"=== Batch {batch_num}/{num_batches} (fixtures {batch_start + 1}-{batch_end}) ===")

        for i, fixture in enumerate(batch, start=batch_start + 1):
            fixture_id = fixture['fixture']['id']

            # Check if already downloaded (resume capability)
            stats_file = f"{RAW_DATA_DIR}/fixture_stats_{fixture_id}.json"
            if os.path.exists(stats_file):
                print(f"[{i}/{len(finished_fixtures)}] Fixture {fixture_id} already downloaded, loading...")
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
            else:
                print(f"[{i}/{len(finished_fixtures)}] Fetching stats for fixture {fixture_id}...")
                stats_data = fetch_fixture_statistics(fixture_id, api_key, api_host)

                if stats_data:
                    # Save raw statistics
                    with open(stats_file, 'w') as f:
                        json.dump(stats_data, f, indent=2)

            if stats_data:
                # Process into rows
                rows = process_fixture_stats(fixture, stats_data)
                all_rows.extend(rows)

        # Wait before next batch (except after the last batch)
        if batch_end < len(finished_fixtures):
            print(f"Waiting {WAIT_TIME} seconds before next batch...")
            time.sleep(WAIT_TIME)
            print()

    # Create DataFrame and save
    print(f"\n=== Complete ===")
    print(f"Processed {len(all_rows)} team-match records")
    df = pd.DataFrame(all_rows)

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{PROCESSED_DATA_DIR}/team_match_stats.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved team match stats to {output_file}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == '__main__':
    main()
