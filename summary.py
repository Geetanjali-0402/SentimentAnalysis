import json
import pandas as pd
import re
from transformers import pipeline
import warnings
from collections import Counter

# FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

# comments from JSON file
with open('comments.json', 'r', encoding='utf-8') as f:
    comments_data = json.load(f)

# DataFrame
comments = []
for item in comments_data['items']:
    comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
    like_count = item['snippet']['topLevelComment']['snippet'].get('likeCount', 0)  # Use get to handle missing 'likeCount'
    comments.append({'comment': comment, 'like_count': like_count})

comments_df = pd.DataFrame(comments)

# Preprocess data
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.lower()  # Convert to lowercase
    return text

comments_df['preprocessed_text'] = comments_df['comment'].apply(preprocess_text)

# sentiment analysis pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

# Add sentiment scores to comments
def get_sentiment_score(text):
    try:
        result = sentiment_pipeline(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']
    except Exception as e:
        print(f"Error processing text: {text}\n{e}")
        return 0.0

comments_df['sentiment_score'] = comments_df['preprocessed_text'].apply(get_sentiment_score)

# Normalize scores to range [-1, 1]
max_score = comments_df['sentiment_score'].max()
min_score = comments_df['sentiment_score'].min()
comments_df['normalized_score'] = comments_df['sentiment_score'].apply(lambda x: 2 * (x - min_score) / (max_score - min_score) - 1)
comments_df['sentiment'] = comments_df['normalized_score'].apply(lambda x: 'positive' if x > 0 else 'negative')

# percentage of positive and negative comments
total_comments = len(comments_df)
positive_comments_count = len(comments_df[comments_df['sentiment'] == 'positive'])
negative_comments_count = len(comments_df[comments_df['sentiment'] == 'negative'])

positive_percentage = (positive_comments_count / total_comments) * 100
negative_percentage = (negative_comments_count / total_comments) * 100

# most liked positive and negative comments
most_liked_positive_comment = comments_df[comments_df['sentiment'] == 'positive'].sort_values(by='like_count', ascending=False).iloc[0]
most_liked_negative_comment = comments_df[comments_df['sentiment'] == 'negative'].sort_values(by='like_count', ascending=False).iloc[0]

def extract_key_themes(comments):
    words = ' '.join(comments).split()
    common_words = Counter(words).most_common(10)
    return common_words

positive_comments = comments_df[comments_df['sentiment'] == 'positive']['preprocessed_text']
negative_comments = comments_df[comments_df['sentiment'] == 'negative']['preprocessed_text']

positive_themes = extract_key_themes(positive_comments)
negative_themes = extract_key_themes(negative_comments)

# Summarize feedback
positive_summary = "Main positive feedback themes:\n" + "\n".join([f"{word}: {count}" for word, count in positive_themes])
negative_summary = "Main negative feedback themes:\n" + "\n".join([f"{word}: {count}" for word, count in negative_themes])

# Display results
print(f"Positive comments: {positive_percentage:.2f}%")
print(f"Negative comments: {negative_percentage:.2f}%\n")

print("Most liked positive comment")
print(f"Comment: {most_liked_positive_comment['comment']}")
print(f"Likes: {most_liked_positive_comment['like_count']}\n")

print("Most liked negative comment:")
print(f"Comment: {most_liked_negative_comment['comment']}")
print(f"Likes: {most_liked_negative_comment['like_count']}\n")