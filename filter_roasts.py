import pandas as pd
import re

# Load the scraped data
df = pd.read_csv("raw_roasts.csv")
# print(f"{df["instruction"]}")

# Define banned words (you can expand this list)
BAD_WORDS = [
    "kill", "die", "suicide", "bomb", "hate", "gun", "religion",
    "gender", "race", "slur1", "slur2", "rape", "murder"
]

def is_safe(text):
    text_low = text.lower()
    return not any(bad in text_low for bad in BAD_WORDS)

# Clean up whitespace, remove unsafe or too-short/long texts
df["roast_text"] = df["roast_text"].apply(lambda t: re.sub(r"\s+", " ", str(t).strip()))
df = df[df["roast_text"].apply(is_safe)]
df = df[df["roast_text"].str.len().between(30, 150)]  # keep witty short ones

# Save filtered CSV
df.to_csv("filtered_roasts.csv", index=False)
print(f"Kept {len(df)} safe roasts out of {len(pd.read_csv('raw_roasts.csv'))}.")

# Convert to JSONL for training
df["instruction"] = "Generate a funny, harmless roast"
df.rename(columns={"roast_text": "response"}, inplace=True)
df[["instruction", "response"]].to_json("roasts_train.jsonl", orient="records", lines=True)

print("Saved filtered_roasts.csv and roasts_train.jsonl")
