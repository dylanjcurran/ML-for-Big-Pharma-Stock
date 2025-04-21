# Reddit_Sent.py
from datetime import datetime
import time
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------
# SETUP
# ----------------------------

reddit = praw.Reddit(
    client_id="bFydw1Neytyth43NixUf9A",
    client_secret="lGsiBCrZssQM5OCDKxHL0ZY6bnccPQ",
    user_agent="reddit_sentiment_analyzer/0.1 by u/dylancurran"
)

analyzer = SentimentIntensityAnalyzer()

# ----------------------------
# FUNCTION
# ----------------------------

def get_reddit_sentiment_custom_window(start_date, end_date, keywords, subreddits, limit=500):
    all_sentiments = []
    total_mentions = 0

    # Convert input window to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(time.mktime(start_dt.timetuple()))
    end_ts = int(time.mktime(end_dt.timetuple()))

    keywords = [k.lower() for k in keywords]

    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        query = " OR ".join(keywords)

        for post in subreddit.search(query=query, sort="new", time_filter="all", limit=limit):
            created = datetime.fromtimestamp(post.created_utc)

            if not (start_dt <= created <= end_dt):
                continue

            text = f"{post.title} {post.selftext}"
            if any(k in text.lower() for k in keywords):
                score = analyzer.polarity_scores(text)["compound"]
                weight = max(post.score, 1)
                all_sentiments.extend([score] * weight)
                total_mentions += 1

            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                comment_created = datetime.fromtimestamp(comment.created_utc)
                if not (start_dt <= comment_created <= end_dt):
                    continue
                if hasattr(comment, "body"):
                    comment_text = comment.body.lower()
                    if any(k in comment_text for k in keywords):
                        c_score = analyzer.polarity_scores(comment.body)["compound"]
                        c_weight = max(comment.score, 1)
                        all_sentiments.extend([c_score] * c_weight)
                        total_mentions += 1

    if all_sentiments:
        return {
            "Reddit Sentiment Score": sum(all_sentiments),
            "Avg Sentiment": sum(all_sentiments) / len(all_sentiments),
            "Sentiment Std": pd.Series(all_sentiments).std(),
            "Number of Mentions": total_mentions,
            "Max Absolute Sentiment": max(abs(s) for s in all_sentiments),
        }
    else:
        return {
            "Reddit Sentiment Score": 0,
            "Avg Sentiment": 0,
            "Sentiment Std": 0,
            "Number of Mentions": 0,
            "Max Absolute Sentiment": 0,
        }

# ----------------------------
# MANUAL TEST
# ----------------------------

if __name__ == "__main__":
    keywords = ["jnj", "johnson", "johnson and johnson", "johnson & johnson", "j&j"]
    subreddits = ["stocks", "investing", "wallstreetbets"]
    time_filter = "month"  # Try "week", "month", or "year"

    result = get_reddit_sentiment_praw(keywords, subreddits, time_filter, limit=250)
    
    # Add context
    result["company"] = "Johnson & Johnson"
    result["time_filter"] = time_filter

    # Save to CSV
    df = pd.DataFrame([result])
    df.to_csv("sentiment_results.csv", index=False)
    print(df)
