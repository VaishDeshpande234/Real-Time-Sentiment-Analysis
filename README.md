# Sentiment Analysis on Twitter Dataset Using Machine Learning

## Project Overview
The goal of this project is to perform sentiment analysis on a large dataset of tweets to classify them as positive or negative. Sentiment analysis helps in understanding the sentiment of users towards specific topics, brands, or events, enabling better decision-making and strategy formulation.

## Dataset
The dataset used in this project is the Sentiment140 dataset, which contains 1,600,000 tweets. Each tweet is labeled as either negative, neutral, or positive.

### Dataset Files
training.1600000.processed.noemoticon.csv: The main dataset containing tweets with sentiment labels.

### Dataset Fields
- sentiment: The polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- ids: The id of the tweet
- date: The date of the tweet
- flag: The query (if any)
- user: The user who tweeted
- text: The text of the tweet

## Project Structure

### Data Preprocessing and Exploratory Data Analysis (EDA)

1. Data Loading: Load the dataset.
2. Data Cleaning: Remove unnecessary columns and replace sentiment values for better understanding (4 to 1 for positive).
3. EDA: Plot the distribution of sentiments and create word clouds for negative and positive tweets.

### Feature Engineering

1. Text Preprocessing:
- Convert text to lowercase.
- Replace URLs, emojis, and usernames with placeholders.
- Remove non-alphanumeric characters and stopwords.
- Lemmatize the words.

2. TF-IDF Vectorization: Convert text data into numerical features using TF-IDF.

### Model Training and Evaluation

1. Train/Test Split: Split the data into training and test sets.
2. Model Training: Train three models - Bernoulli Naive Bayes, LinearSVC, and Logistic Regression.
3. Model Evaluation: Evaluate models using precision, recall, f1-score, and confusion matrix.

### Results
The models were evaluated based on precision, recall, f1-score, and confusion matrix:

1. Bernoulli Naive Bayes:

- Precision: 0.81 (Negative), 0.80 (Positive)
- Recall: 0.79 (Negative), 0.81 (Positive)
- F1-Score: 0.80 (Negative), 0.80 (Positive)
- Accuracy: 0.80

2. LinearSVC:

- Precision: 0.82 (Negative), 0.81 (Positive)
- Recall: 0.81 (Negative), 0.83 (Positive)
- F1-Score: 0.82 (Negative), 0.82 (Positive)
- Accuracy: 0.82

3. Logistic Regression:

- Precision: 0.83 (Negative), 0.82 (Positive)
- Recall: 0.82 (Negative), 0.84 (Positive)
- F1-Score: 0.83 (Negative), 0.83 (Positive)
- Accuracy: 0.83

### Visualization

1. Word Cloud for Negative Tweets: Visual representation of the most frequent words in negative tweets.

2. Word Cloud for Positive Tweets: Visual representation of the most frequent words in positive tweets.

3. Confusion Matrix: Heatmap of the confusion matrix showing the performance of the models in terms of true positives, false positives, false negatives, and true negatives.

### Saving and Loading Models

- Model Saving: Save the trained models and vectorizer using pickle for future use.
- Model Loading: Load the saved models and vectorizer to make predictions on new data.

## Conclusion
This project demonstrates a comprehensive approach to performing sentiment analysis on Twitter data using machine learning. The models trained provide good accuracy and can be used for real-time sentiment analysis applications.
