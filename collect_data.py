"""
Reddit Data Collector using PRAW
---------------------------------
Collects posts from specified subreddits and saves to CSV.

Usage:
    Set your Reddit API credentials as environment variables:
        export REDDIT_CLIENT_ID=your_client_id
        export REDDIT_CLIENT_SECRET=your_client_secret
        export REDDIT_USER_AGENT=reddit-sentiment-scraper/1.0

    Then run:
        python collect_data.py

    To get credentials:
        1. Go to https://www.reddit.com/prefs/apps
        2. Create a new app (type: script)
        3. Copy client_id and client_secret
"""

import os
import praw
import pandas as pd
from datetime import datetime

# --- Config ---
SUBREDDITS = ["ecommerce", "korea", "technology", "logistics", "investing"]
POST_LIMIT = 100   # posts per subreddit
SORT_BY    = "hot"  # "hot" | "new" | "top"
OUTPUT     = "data/reddit_posts.csv"

def collect(reddit, subreddit_name, limit=100, sort="hot"):
    sub = reddit.subreddit(subreddit_name)
    method = getattr(sub, sort)
    rows = []
    for post in method(limit=limit):
        rows.append({
            "id":           post.id,
            "subreddit":    subreddit_name,
            "title":        post.title,
            "text":         post.selftext,
            "score":        post.score,
            "num_comments": post.num_comments,
            "created_utc":  int(post.created_utc),
        })
    return rows


def main():
    client_id     = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent    = os.environ.get("REDDIT_USER_AGENT", "reddit-sentiment-scraper/1.0")

    if not client_id or not client_secret:
        print("ERROR: Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        print("       See the docstring at the top of this file for instructions.")
        return

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    all_rows = []
    for sub in SUBREDDITS:
        print(f"Collecting r/{sub} ...")
        rows = collect(reddit, sub, limit=POST_LIMIT, sort=SORT_BY)
        all_rows.extend(rows)
        print(f"  → {len(rows)} posts")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} posts to {OUTPUT}")


if __name__ == "__main__":
    main()
