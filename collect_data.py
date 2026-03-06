"""
Reddit Data Collector
---------------------
Two collection methods:

Method 1 (default) — Reddit public JSON API, NO credentials required:
    python collect_data.py

Method 2 — PRAW (more posts, private subreddits, search):
    export REDDIT_CLIENT_ID=your_id
    export REDDIT_CLIENT_SECRET=your_secret
    python collect_data.py --praw

Get PRAW credentials: https://www.reddit.com/prefs/apps (create a "script" app)
"""

import os
import time
import argparse
import requests
import pandas as pd

# --- Config ---
SUBREDDITS = {
    "ecommerce":       ("top", "year"),
    "korea":           ("top", "year"),
    "logistics":       ("top", "all"),
    "investing":       ("top", "month"),
    "MachineLearning": ("top", "year"),
    "supplychain":     ("top", "all"),
    "startups":        ("top", "month"),
}
POST_LIMIT = 100
OUTPUT = "data/real_reddit_posts.csv"
HEADERS = {"User-Agent": "portfolio-sentiment-analysis/1.0 (research)"}


# ── Method 1: Reddit public JSON API ──────────────────────────────────────────

def fetch_json_api(subreddit, sort="top", time_filter="year", limit=100):
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&t={time_filter}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"  ✗ r/{subreddit}: HTTP {r.status_code}")
        return []
    rows = []
    for p in r.json()["data"]["children"]:
        d = p["data"]
        if d.get("stickied"):
            continue
        text = d.get("selftext", "")
        if text in ("", "[deleted]", "[removed]"):
            continue
        rows.append({
            "id":           d["id"],
            "subreddit":    subreddit,
            "title":        d.get("title", ""),
            "text":         text,
            "score":        d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "created_utc":  int(d.get("created_utc", 0)),
        })
    return rows


def collect_json_api():
    print("Method: Reddit public JSON API (no credentials)\n")
    all_rows = []
    for sub, (sort, period) in SUBREDDITS.items():
        rows = fetch_json_api(sub, sort=sort, time_filter=period, limit=POST_LIMIT)
        all_rows.extend(rows)
        print(f"  r/{sub}: {len(rows)} posts")
        time.sleep(1.2)
    return all_rows


# ── Method 2: PRAW ─────────────────────────────────────────────────────────────

def collect_praw():
    import praw

    client_id     = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent    = os.environ.get("REDDIT_USER_AGENT", "portfolio-sentiment-analysis/1.0")

    if not client_id or not client_secret:
        print("ERROR: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set.")
        return []

    print("Method: PRAW (authenticated)\n")
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                         user_agent=user_agent)
    all_rows = []
    for sub, (sort, period) in SUBREDDITS.items():
        subreddit = reddit.subreddit(sub)
        method = getattr(subreddit, sort)
        kwargs = {"limit": POST_LIMIT}
        if sort in ("top", "controversial"):
            kwargs["time_filter"] = period
        rows = []
        for post in method(**kwargs):
            if post.stickied or post.selftext in ("", "[deleted]", "[removed]"):
                continue
            rows.append({
                "id":           post.id,
                "subreddit":    sub,
                "title":        post.title,
                "text":         post.selftext,
                "score":        post.score,
                "num_comments": post.num_comments,
                "created_utc":  int(post.created_utc),
            })
        all_rows.extend(rows)
        print(f"  r/{sub}: {len(rows)} posts")
    return all_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--praw", action="store_true",
                        help="Use PRAW instead of JSON API (requires credentials)")
    args = parser.parse_args()

    rows = collect_praw() if args.praw else collect_json_api()

    if not rows:
        print("No data collected.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    df = df[df["title"].str.len() > 10]
    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved {len(df)} posts → {OUTPUT}")
    print(df.groupby("subreddit").size().to_string())


if __name__ == "__main__":
    main()
