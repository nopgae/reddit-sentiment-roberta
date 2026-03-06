"""Generate visualization PNGs for reddit-sentiment-roberta."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs('images', exist_ok=True)

df = pd.read_csv('data/real_reddit_posts.csv')

PALETTE = {'positive': '#38A169', 'neutral': '#718096', 'negative': '#E53E3E'}
LABEL_ORDER = ['positive', 'neutral', 'negative']

# ── Chart 1: Sentiment Distribution (donut) ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

counts = df['sentiment'].value_counts()
ordered_labels = [l for l in LABEL_ORDER if l in counts.index]
sizes  = [counts[l] for l in ordered_labels]
colors = [PALETTE[l] for l in ordered_labels]
pcts   = [f'{s/len(df)*100:.1f}%' for s in sizes]
labels = [f'{l.capitalize()}\n{p}\n({s} posts)'
          for l, p, s in zip(ordered_labels, pcts, sizes)]

wedges, texts = ax.pie(
    sizes, labels=labels, colors=colors,
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops={'fontsize': 13}
)
for t in texts:
    t.set_fontweight('bold')

ax.set_title('Overall Sentiment Distribution\n563 Real Reddit Posts (7 Subreddits)',
             fontsize=17, fontweight='bold', pad=20)

avg_score = df['sentiment_score'].mean()
ax.text(0, 0, f'Avg Score\n{avg_score:.3f}',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#2D3748')

patches = [mpatches.Patch(color=PALETTE[l], label=l.capitalize()) for l in LABEL_ORDER]
ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig('images/sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('✓ sentiment_distribution.png')

# ── Chart 2: Sentiment by Subreddit (stacked bar) ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

sub_counts = df.groupby(['subreddit', 'sentiment']).size().unstack(fill_value=0)
# Reorder columns
for col in LABEL_ORDER:
    if col not in sub_counts.columns:
        sub_counts[col] = 0
sub_counts = sub_counts[LABEL_ORDER]

# Sort subreddits by net sentiment score (positive - negative)
sub_score = df.groupby('subreddit')['sentiment_score'].mean().sort_values(ascending=False)
sub_counts = sub_counts.loc[sub_score.index]

bottom_pos = np.zeros(len(sub_counts))
bars = {}
for sentiment in LABEL_ORDER:
    vals = sub_counts[sentiment].values
    b = ax.bar(sub_counts.index, vals, bottom=bottom_pos,
               color=PALETTE[sentiment], label=sentiment.capitalize(),
               edgecolor='white', linewidth=0.8, width=0.65)
    bars[sentiment] = b
    # Add count labels inside bars
    for i, (v, bot) in enumerate(zip(vals, bottom_pos)):
        if v >= 5:
            ax.text(i, bot + v / 2, str(v),
                    ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold')
    bottom_pos += vals

# Add avg sentiment score annotation above each bar
totals = sub_counts.sum(axis=1)
for i, sub in enumerate(sub_counts.index):
    score = sub_score[sub]
    ax.text(i, totals[sub] + 1, f'{score:+.2f}',
            ha='center', va='bottom', fontsize=11, color='#2D3748', fontweight='bold')

ax.set_xlabel('Subreddit', fontsize=13, labelpad=10)
ax.set_ylabel('Number of Posts', fontsize=13, labelpad=10)
ax.set_title('Sentiment Distribution by Subreddit\n(number above bar = avg sentiment score)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticklabels([f'r/{s}' for s in sub_counts.index], fontsize=11, rotation=20, ha='right')
ax.tick_params(axis='y', labelsize=10)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.set_ylim(0, totals.max() * 1.15)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('images/sentiment_by_subreddit.png', dpi=150, bbox_inches='tight')
plt.close()
print('✓ sentiment_by_subreddit.png')

# ── Chart 3: Confidence box plot by sentiment ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

data_by_sentiment = [df[df['sentiment'] == s]['confidence'].values for s in LABEL_ORDER]
bp = ax.boxplot(data_by_sentiment, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, sentiment in zip(bp['boxes'], LABEL_ORDER):
    patch.set_facecolor(PALETTE[sentiment])
    patch.set_alpha(0.85)

means = [df[df['sentiment'] == s]['confidence'].mean() for s in LABEL_ORDER]
for i, mean in enumerate(means, 1):
    ax.plot(i, mean, 'D', color='white', markersize=8, zorder=5)
    ax.text(i + 0.15, mean, f'μ={mean:.3f}', va='center', fontsize=11, color='#2D3748')

ax.set_xticks([1, 2, 3])
ax.set_xticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=13)
ax.set_ylabel('Model Confidence Score', fontsize=13, labelpad=10)
ax.set_xlabel('Sentiment Class', fontsize=13, labelpad=10)
ax.set_title('RoBERTa Model Confidence by Sentiment Class\n(diamond = mean, white line = median)',
             fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_ylim(0.3, 1.02)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('images/confidence_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print('✓ confidence_by_sentiment.png')

print('\nAll charts saved to images/')
