# SentimentAnalysis
This project aims to extract sentiments from YouTube comments using a pre-trained sentiment analysis model from the Hugging Face Transformers library. It preprocesses the comments, performs sentiment analysis, and normalizes the sentiment scores.

1. Data Collection: YouTube comments are obtained from the YouTube Data API or any other source that provides access to YouTube comments in JSON format. These comments are stored in a JSON file (comments.json) for processing.

2. Preprocessing: The comments undergo preprocessing to clean and standardize the text data. This involves: Removing extra spaces, URLs, mentions, and hashtags. Converting text to lowercase to ensure consistency.

3. Sentiment Analysis: The preprocessed comments are fed into a pre-trained sentiment analysis model. The model assigns sentiment scores to each comment, indicating the degree of positivity or negativity. The sentiment scores are obtained using the BERT-based model, which has been fine-tuned on sentiment analysis tasks and is capable of understanding nuances in language.

4. Displaying Results: The comments along with their corresponding normalized sentiment scores are printed to the console. This allows users to quickly understand the sentiment expressed in each comment without manual inspection.
