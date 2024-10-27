import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import string
from collections import Counter
import psutil
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration parameters
USE_FULL_DATASET = True  # Set to True to use full dataset, False to use sampling
SAMPLE_SIZE = 100000     # Only used if USE_FULL_DATASET = False
TEST_SAMPLE_SIZE = 20000 # Only used if USE_FULL_DATASET = False
BATCH_SIZE = 50000      # Batch size for processing full dataset
RANDOM_SEED = 42

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def print_memory_usage(message=""):
    """Print current memory usage"""
    memory_gb = get_memory_usage()
    print(f"Memory usage {message}: {memory_gb:.2f} GB")

class BatchDataProcessor:
    def __init__(self, file_path, batch_size=50000):
        self.file_path = file_path
        self.batch_size = batch_size
        self.total_rows = sum(1 for _ in open(file_path)) - 1
        print(f"Total rows in {os.path.basename(file_path)}: {self.total_rows:,}")
    
    def process_in_batches(self, processing_func, accumulator_func=None):
        """Process file in batches with optional accumulation of results"""
        results = []
        processed_rows = 0
        
        for chunk in pd.read_csv(self.file_path, chunksize=self.batch_size):
            processed_rows += len(chunk)
            print(f"\nProcessing batch... Progress: {processed_rows:,}/{self.total_rows:,} rows")
            print_memory_usage("before processing")
            
            processed_chunk = processing_func(chunk)
            
            if accumulator_func:
                accumulator_func(processed_chunk)
            else:
                results.append(processed_chunk)
            
            print_memory_usage("after processing")
        
        if results:
            return pd.concat(results, ignore_index=True)

def extract_advanced_text_features(text):
    """Extract advanced features from text"""
    if not isinstance(text, str):
        return {
            'sentiment_score': 0,
            'subjectivity': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'capital_word_ratio': 0,
            'avg_word_length': 0,
            'unique_word_ratio': 0
        }
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # TextBlob analysis
    blob = TextBlob(text)
    
    # Special characters
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Word statistics
    words = text.split()
    if not words:
        return {
            'sentiment_score': sentiment_scores['compound'],
            'subjectivity': 0,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_word_ratio': 0,
            'avg_word_length': 0,
            'unique_word_ratio': 0
        }
    
    capital_words = sum(1 for word in words if word.isupper())
    avg_word_length = sum(len(word) for word in words) / len(words)
    unique_words = len(set(words))
    
    return {
        'sentiment_score': sentiment_scores['compound'],
        'subjectivity': blob.sentiment.subjectivity,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'capital_word_ratio': capital_words / len(words),
        'avg_word_length': avg_word_length,
        'unique_word_ratio': unique_words / len(words)
    }

def clean_text(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation but keep important markers
    punct = string.punctuation.replace('!', '').replace('?', '')
    text = ''.join(char for char in text if char not in punct)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def add_features_to(df):
    """Enhanced feature engineering"""
    print("Adding features...")
    
    # Basic features
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    
    # Time-based features
    df['DateTime'] = pd.to_datetime(df['Time'], unit='s')
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Hour'] = df['DateTime'].dt.hour
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # User engagement features
    df['HasHelpfulnessVotes'] = (df['HelpfulnessDenominator'] > 0).astype(int)
    df['HelpfulnessRatio'] = df['Helpfulness']
    df['IsHighlyHelpful'] = (df['Helpfulness'] > 0.8).astype(int)
    df['VoteCount'] = df['HelpfulnessDenominator']
    df['HighVoteCount'] = (df['VoteCount'] > df['VoteCount'].median()).astype(int)
    
    # Text cleaning
    df['cleaned_summary'] = df['Summary'].fillna('').apply(clean_text)
    df['cleaned_text'] = df['Text'].fillna('').apply(clean_text)
    
    # Text length features
    df['summary_length'] = df['Summary'].fillna('').str.len()
    df['text_length'] = df['Text'].fillna('').str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['avg_word_length'] = df['cleaned_text'].apply(lambda x: np.mean([len(word) for word in x.split()] or [0]))
    
    # Extract advanced text features
    print("Extracting advanced text features...")
    text_features = df['cleaned_text'].apply(extract_advanced_text_features)
    for feature, values in zip(text_features.iloc[0].keys(), zip(*text_features.values)):
        df[f'text_{feature}'] = values
    
    # TF-IDF features
    print("Calculating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5
    )
    text_features = tfidf.fit_transform(df['cleaned_text'])
    text_features_df = pd.DataFrame(
        text_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(200)]
    )
    
    return pd.concat([df, text_features_df], axis=1)

def process_data_with_sampling():
    """Process data using sampling approach"""
    print("\nProcessing data with sampling...")
    
    n_rows = sum(1 for _ in open("./data/train.csv")) - 1
    skip_rows = sorted(random.sample(range(1, n_rows + 1), n_rows - SAMPLE_SIZE))
    trainingSet = pd.read_csv("./data/train.csv", skiprows=skip_rows)
    
    n_rows_test = sum(1 for _ in open("./data/test.csv")) - 1
    skip_rows_test = sorted(random.sample(range(1, n_rows_test + 1), 
                                        n_rows_test - TEST_SAMPLE_SIZE))
    testingSet = pd.read_csv("./data/test.csv", skiprows=skip_rows_test)
    
    train = add_features_to(trainingSet)
    X_submission = pd.merge(train, testingSet, left_on='Id', right_on='Id')
    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})
    X_train = train[train['Score'].notnull()]
    
    return X_train, X_submission

def process_full_dataset():
    """Process the full dataset using batch processing"""
    print("\nProcessing full dataset in batches...")
    
    train_processor = BatchDataProcessor("./data/train.csv", BATCH_SIZE)
    test_processor = BatchDataProcessor("./data/test.csv", BATCH_SIZE)
    
    train_data = train_processor.process_in_batches(add_features_to)
    test_data = test_processor.process_in_batches(lambda df: df)
    
    X_submission = pd.merge(train_data, test_data, left_on='Id', right_on='Id')
    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})
    X_train = train_data[train_data['Score'].notnull()]
    
    return X_train, X_submission

def train_model(X_train_select, Y_train, X_test_select, Y_test, X_submission_select):
    """Train the improved model"""
    print("\nTraining models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_select)
    X_test_scaled = scaler.transform(X_test_select)
    X_submission_scaled = scaler.transform(X_submission_select)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_SEED
    )
    
    # Train models
    print("Training Random Forest...")
    rf_model.fit(X_train_scaled, Y_train)
    
    print("Training Gradient Boosting...")
    gb_model.fit(X_train_scaled, Y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    final_pred = np.round((rf_pred + gb_pred) / 2).astype(int)
    
    # Evaluate
    print("\nModel Performance:")
    print("Random Forest Accuracy:", accuracy_score(Y_test, rf_pred))
    print("Gradient Boosting Accuracy:", accuracy_score(Y_test, gb_pred))
    print("Ensemble Accuracy:", accuracy_score(Y_test, final_pred))
    print("\nClassification Report:")
    print(classification_report(Y_test, final_pred))
    
    # Create submission predictions
    rf_submission = rf_model.predict(X_submission_scaled)
    gb_submission = gb_model.predict(X_submission_scaled)
    final_submission = np.round((rf_submission + gb_submission) / 2).astype(int)
    
    return final_submission

def main():
    # Download NLTK data
    print("Downloading NLTK data...")
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
    print_memory_usage("at start")
    
    # Define features
    features = [
        # Basic features
        'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Helpfulness',
        'Year', 'Month', 'DayOfWeek', 'Hour', 'IsWeekend',
        
        # User engagement features
        'HasHelpfulnessVotes', 'HelpfulnessRatio', 'IsHighlyHelpful',
        'VoteCount', 'HighVoteCount',
        
        # Text features
        'summary_length', 'text_length', 'word_count', 'avg_word_length',
        
        # Sentiment features
        'text_sentiment_score', 'text_subjectivity',
        'text_exclamation_count', 'text_question_count',
        'text_capital_word_ratio', 'text_avg_word_length',
        'text_unique_word_ratio'
    ] + [f'tfidf_{i}' for i in range(200)]
    
    # Process data
    if USE_FULL_DATASET:
        X_train, X_submission = process_full_dataset()
    else:
        X_train, X_submission = process_data_with_sampling()
    
    # Split data
    print("\nSplitting data...")
    X_train_split, X_test_split, Y_train, Y_test = train_test_split(
        X_train.drop(columns=['Score']),
        X_train['Score'],
        test_size=0.25,
        random_state=RANDOM_SEED
    )
    
    # Select features
    X_train_select = X_train_split[features]
    X_test_select = X_test_split[features]
    X_submission_select = X_submission[features]
    
    # Train model and make predictions
    final_predictions = train_model(
        X_train_select, Y_train, 
        X_test_select, Y_test,
        X_submission_select
    )
    
    # Create submission file
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'Id': X_submission['Id'],
        'Score': final_predictions
    })
    submission.to_csv("./data/submission.csv", index=False)
    print("Submission file created!")
    
    print_memory_usage("at end")

if __name__ == "__main__":
    import random
    random.seed(RANDOM_SEED)
    main()