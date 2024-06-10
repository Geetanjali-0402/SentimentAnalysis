import pandas as pd
import re
import json
import nltk
import warnings
from sklearn.model_selection import train_test_split
from transformers import pipeline

# comments from JSON file
with open('comments.json', 'r', encoding='utf-8') as f:
    comments_data = json.load(f)

# DataFrame from the comments
comments = [item['snippet']['topLevelComment']['snippet']['textOriginal'] for item in comments_data['items']]
comments_df = pd.DataFrame(comments, columns=['comment'])

# Preprocess data
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.lower()  # Convert to lowercase
    return text

comments_df['preprocessed_text'] = comments_df['comment'].apply(preprocess_text)

# FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

# sentiment analysis pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

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

for i, row in comments_df.iterrows():
    print(f"Comment {i+1}: {row['comment']} - Sentiment Score: \033[1m{row['normalized_score']}\033[0m")

train_texts, val_texts = train_test_split(comments_df['preprocessed_text'], test_size=0.2, random_state=42)