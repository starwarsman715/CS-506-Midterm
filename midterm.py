import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import nltk
from datetime import datetime
import warnings
import psutil
import os
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
        
        for chunk in pd.read_csv(self.file_path, chunksize=self.batch_size):
            print(f"\nProcessing batch of {len(chunk):,} rows...")
            print_memory_usage("before processing")
            
            # Process the chunk
            processed_chunk = processing_func(chunk)
            
            # Either accumulate results or yield processed chunk
            if accumulator_func:
                accumulator_func(processed_chunk)
            else:
                results.append(processed_chunk)
            
            print_memory_usage("after processing")
        
        if results:
            return pd.concat(results, ignore_index=True)

def clean_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

def extract_text_features(df):
    """Extract features from text columns"""
    print("Extracting text features...")
    
    # Clean text
    df['cleaned_summary'] = df['Summary'].fillna('').apply(clean_text)
    df['cleaned_text'] = df['Text'].fillna('').apply(clean_text)
    
    # Text length features
    df['summary_length'] = df['Summary'].fillna('').str.len()
    df['text_length'] = df['Text'].fillna('').str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    text_features = tfidf.fit_transform(df['cleaned_text'])
    
    text_features_df = pd.DataFrame(
        text_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(100)]
    )
    
    return pd.concat([df, text_features_df], axis=1)

def add_features_to(df):
    """Add all features to the dataset"""
    print("Adding features...")
    
    # Basic feature engineering
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    
    # Convert timestamp to datetime features
    df['DateTime'] = pd.to_datetime(df['Time'], unit='s')
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    # User engagement features
    df['HasHelpfulnessVotes'] = (df['HelpfulnessDenominator'] > 0).astype(int)
    df['HelpfulnessRatio'] = df['Helpfulness']
    df['IsHighlyHelpful'] = (df['Helpfulness'] > 0.8).astype(int)
    
    # Extract text features
    df = extract_text_features(df)
    
    return df

def process_data_with_sampling():
    """Process data using sampling approach"""
    print("\nProcessing data with sampling...")
    
    # Load sampled data
    n_rows = sum(1 for _ in open("./data/train.csv")) - 1
    skip_rows = sorted(random.sample(range(1, n_rows + 1), n_rows - SAMPLE_SIZE))
    trainingSet = pd.read_csv("./data/train.csv", skiprows=skip_rows)
    
    n_rows_test = sum(1 for _ in open("./data/test.csv")) - 1
    skip_rows_test = sorted(random.sample(range(1, n_rows_test + 1), 
                                        n_rows_test - TEST_SAMPLE_SIZE))
    testingSet = pd.read_csv("./data/test.csv", skiprows=skip_rows_test)
    
    # Process the data
    train = add_features_to(trainingSet)
    X_submission = pd.merge(train, testingSet, left_on='Id', right_on='Id')
    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})
    X_train = train[train['Score'].notnull()]
    
    return X_train, X_submission

def process_full_dataset():
    """Process the full dataset using batch processing"""
    print("\nProcessing full dataset in batches...")
    
    # Initialize batch processors
    train_processor = BatchDataProcessor("./data/train.csv", BATCH_SIZE)
    test_processor = BatchDataProcessor("./data/test.csv", BATCH_SIZE)
    
    # Process training data in batches
    train_data = train_processor.process_in_batches(add_features_to)
    test_data = test_processor.process_in_batches(lambda df: df)
    
    # Merge and prepare final datasets
    X_submission = pd.merge(train_data, test_data, left_on='Id', right_on='Id')
    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})
    X_train = train_data[train_data['Score'].notnull()]
    
    return X_train, X_submission

def train_and_evaluate_model(X_train, X_submission, features):
    """Train model and create predictions"""
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
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    print_memory_usage("before training")
    model.fit(X_train_select, Y_train)
    print_memory_usage("after training")
    
    # Make predictions
    Y_test_predictions = model.predict(X_test_select)
    
    # Evaluate model
    print(f"\nAccuracy on testing set: {accuracy_score(Y_test, Y_test_predictions):.4f}")
    
    # Create submission predictions
    X_submission['Score'] = model.predict(X_submission_select)
    submission = X_submission[['Id', 'Score']]
    submission.to_csv("./data/submission.csv", index=False)
    print("\nSubmission file created!")
    
    return model, Y_test, Y_test_predictions

def main():
    print_memory_usage("at start")
    
    # Define features
    features = [
        'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Helpfulness',
        'Year', 'Month', 'DayOfWeek',
        'HasHelpfulnessVotes', 'HelpfulnessRatio', 'IsHighlyHelpful',
        'summary_length', 'text_length', 'word_count'
    ] + [f'tfidf_{i}' for i in range(100)]
    
    # Process data based on configuration
    if USE_FULL_DATASET:
        X_train, X_submission = process_full_dataset()
    else:
        X_train, X_submission = process_data_with_sampling()
    
    # Train model and make predictions
    model, Y_test, Y_test_predictions = train_and_evaluate_model(
        X_train, X_submission, features
    )
    
    print_memory_usage("at end")

if __name__ == "__main__":
    import random
    random.seed(RANDOM_SEED)
    
    # Download NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
    main()