# Reddit Sentiment Analysis with RoBERTa

End-to-end NLP pipeline that classifies sentiment of **563 real Reddit posts** using a pre-trained RoBERTa transformer. Covers e-commerce, Korean tech, logistics, supply chain, and investing discussions — directly relevant to data science and SCM roles.

## Tech Stack

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa-base fine-tuned on 124M tweets)
- **Framework:** PyTorch + Hugging Face Transformers
- **Data collection:** Reddit public JSON API (no credentials) + PRAW (optional)
- **Analysis & Visualization:** pandas, Plotly

## Pipeline

```
Reddit JSON API (no auth required)
           ↓
  563 real posts across 7 subreddits
           ↓
  Text Preprocessing   ← strip markdown, URLs, user mentions
           ↓
  RoBERTa Inference    ← Positive / Neutral / Negative + confidence score
           ↓
  Analysis & Plots     ← distribution, subreddit comparison, trends, keywords
```

## Project Structure

```
.
├── reddit_sentiment_roberta.ipynb   # Main analysis notebook
├── collect_data.py                  # Data collection (JSON API + PRAW)
├── data/
│   ├── real_reddit_posts.csv       # 563 real posts with RoBERTa labels
│   └── sample_posts.csv            # 50-post fallback (offline use)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook reddit_sentiment_roberta.ipynb
```

Runs on `data/real_reddit_posts.csv` — 563 actual Reddit posts already labeled by RoBERTa. No API key needed to view analysis.

## Re-collecting Fresh Data

```bash
# No credentials required (Reddit public JSON API)
python collect_data.py

# With PRAW (more posts, full search capability)
export REDDIT_CLIENT_ID=your_id
export REDDIT_CLIENT_SECRET=your_secret
python collect_data.py --praw
```

## Analyses Included

| Section | What it shows |
|---|---|
| Sentiment Distribution | Overall Positive / Neutral / Negative donut chart |
| By Subreddit | Stacked bar — which communities skew most positive/negative |
| Temporal Trend | Weekly sentiment counts + 7-post rolling mean score |
| Top Posts | Highest-upvoted posts per sentiment class |
| Confidence Analysis | Box plot — model certainty per class |
| Score vs Sentiment | Scatter — Reddit upvotes × confidence × comment volume |
| Keyword Analysis | Most frequent words per sentiment class |

## Results (563 real posts)

```
neutral  : 267 posts (47.4%)  avg confidence = 0.681
negative : 188 posts (33.4%)  avg confidence = 0.683
positive : 108 posts (19.2%)  avg confidence = 0.748

Overall sentiment score : -0.142 (slightly negative — realistic for complaint-heavy subs)
Most positive subreddit : r/supplychain
Most negative subreddit : r/ecommerce
```

## Dataset

| Subreddit | Posts | Topic |
|---|---|---|
| r/ecommerce | 99 | Online retail, platforms, fulfillment |
| r/investing | 100 | Markets, stocks, macro |
| r/MachineLearning | 98 | AI/ML research and industry |
| r/startups | 100 | Entrepreneurship, funding, growth |
| r/supplychain | 74 | Logistics, procurement, operations |
| r/korea | 54 | Korean society, economy, tech |
| r/logistics | 38 | Last-mile, warehousing, transport |

## Why RoBERTa?

- Twitter-trained RoBERTa generalizes well to Reddit — both are short, opinionated social media text
- 3-class output (Positive / Neutral / Negative) captures nuance better than binary sentiment
- Confidence scores enable filtering uncertain predictions for downstream use

## Relevance to Data Science / SCM Roles

- **Customer sentiment monitoring** — product and brand perception in e-commerce pipelines
- **Demand signal extraction** — community sentiment as a leading indicator for inventory planning
- **NLP pipeline design** — preprocessing → batched inference → aggregation pattern used in production
- **Transformer deployment** — efficient batching and serving of HuggingFace models

---

*Data collected via Reddit public JSON API · Model: cardiffnlp/twitter-roberta-base-sentiment-latest*
