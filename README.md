# Reddit Sentiment Analysis with RoBERTa

End-to-end NLP pipeline for classifying Reddit post sentiment using a pre-trained RoBERTa transformer model. Analyzes discussions across **e-commerce, Korean tech, logistics, and investing** subreddits — topics directly relevant to supply chain and data science roles.

## Tech Stack

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa-base fine-tuned on 124M tweets)
- **Framework:** PyTorch + Hugging Face Transformers
- **Data collection:** PRAW (Reddit API wrapper)
- **Analysis & Visualization:** pandas, Plotly

## Pipeline

```
Reddit API (PRAW)
      ↓
Raw Posts CSV
      ↓
Text Preprocessing   ← strip markdown, URLs, mentions
      ↓
RoBERTa Inference    ← Positive / Neutral / Negative + confidence score
      ↓
Analysis & Plots     ← distribution, trends, keywords, score correlation
```

## Project Structure

```
.
├── reddit_sentiment_roberta.ipynb   # Main analysis notebook
├── collect_data.py                  # PRAW data collection script
├── data/
│   └── sample_posts.csv            # 50 curated posts (runs without API credentials)
├── requirements.txt
└── README.md
```

## Quick Start (no API key needed)

```bash
pip install -r requirements.txt
jupyter notebook reddit_sentiment_roberta.ipynb
```

The notebook runs on `data/sample_posts.csv` by default — 50 posts across 5 subreddits with realistic e-commerce/tech discussions.

## Collecting Live Reddit Data

```bash
# 1. Get Reddit API credentials at https://www.reddit.com/prefs/apps
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret

# 2. Collect posts
python collect_data.py   # saves to data/reddit_posts.csv

# 3. In the notebook, change DATA_PATH to 'data/reddit_posts.csv'
```

## Analyses Included

| Section | What it shows |
|---|---|
| Sentiment Distribution | Overall Positive / Neutral / Negative breakdown (donut chart) |
| By Subreddit | Stacked bar — which communities are most positive/negative |
| Temporal Trend | Weekly sentiment counts + 7-post rolling mean sentiment score |
| Top Posts | Highest-scored posts by sentiment class |
| Confidence Analysis | Box plot of model certainty per class |
| Score vs Sentiment | Scatter — Reddit upvotes vs model confidence vs comment volume |
| Keyword Analysis | Most frequent words per sentiment class (bar charts) |

## Sample Results (50 posts)

```
positive :  23 posts (46%)  avg confidence = 0.811
negative :  18 posts (36%)  avg confidence = 0.835
neutral  :   9 posts (18%)  avg confidence = 0.561

Overall sentiment score: +0.10 (slightly positive)
Most positive subreddit: r/technology
Most negative subreddit: r/ecommerce
```

## Why RoBERTa for Reddit?

- Twitter-trained RoBERTa generalizes well to Reddit — both are short, opinionated social media text
- 3-class labels (Positive / Neutral / Negative) are more informative than binary for nuanced discussions
- Confidence scores enable filtering low-certainty predictions for downstream use

## Relevance to Data Science / SCM Roles

- **Customer sentiment monitoring** — product reviews, brand perception tracking in e-commerce
- **Demand signal extraction** — sentiment on product categories as a leading indicator
- **NLP pipeline design** — preprocessing → inference → aggregation pattern used in production systems
- **Transformer model deployment** — loading, batching, and serving HuggingFace models efficiently

---

*Subreddits: r/ecommerce · r/korea · r/technology · r/logistics · r/investing*
