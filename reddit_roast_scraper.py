import praw
import pandas as pd
from tqdm import tqdm

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="fAt5xdw4FOGb_81MbrOd2Q",
    client_secret="afmKqY0lHC2ke8knq2EIIf61cbFyVA",
    user_agent="roast_collector by u/magnasspark",
)

subreddit = reddit.subreddit("RoastMe")

data = []

for post in tqdm(subreddit.hot(limit=200)):  # you can use .top(), .new()
    post.comments.replace_more(limit=0)
    for comment in post.comments:
        if comment.body and len(comment.body) > 20:
            data.append({
                "post_title": post.title,
                "roast_text": comment.body
            })

df = pd.DataFrame(data)
df.to_csv("raw_roasts.csv", index=False)
print(f"Saved {len(df)} roasts.")
