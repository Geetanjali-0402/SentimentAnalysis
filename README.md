# SentimentAnalysis
This project aims to extract sentiments from YouTube comments using a pre-trained sentiment analysis model from the Hugging Face Transformers library. It preprocesses the comments, performs sentiment analysis, and normalizes the sentiment scores.

1. Data Collection: YouTube comments are obtained from the YouTube Data API or any other source that provides access to YouTube comments in JSON format. These comments are stored in a JSON file (comments.json) for processing.

2. Preprocessing: The comments undergo preprocessing to clean and standardize the text data. This involves: Removing extra spaces, URLs, mentions, and hashtags. Converting text to lowercase to ensure consistency.
