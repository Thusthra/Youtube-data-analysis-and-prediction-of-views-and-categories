"""
Data Preprocessing Module for YouTube Data Analysis
Handles data cleaning, feature engineering, and text processing
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataPreprocessor:
    """Preprocesses YouTube video data for analysis and modeling"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, df):
        """Main preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Convert data types
        df = self._convert_data_types(df)
        
        # Clean text data
        df = self._clean_text_data(df)
        
        # Create features
        df = self._create_features(df)
        
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing/null values"""
        logger.info("Handling missing values...")
        
        # Fill numeric columns with median
        numeric_cols = ['view_count', 'like_count', 'comment_count', 'favorite_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = ['category_name', 'category_id', 'channel_title']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Fill text columns with empty string
        text_cols = ['title', 'description', 'tags']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate videos"""
        initial_count = len(df)
        
        if 'video_id' in df.columns:
            df = df.drop_duplicates(subset=['video_id'], keep='first')
        else:
            df = df.drop_duplicates()
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")
        
        return df
    
    def _convert_data_types(self, df):
        """Convert data types to appropriate formats"""
        logger.info("Converting data types...")
        
        # Convert numeric columns
        numeric_cols = ['view_count', 'like_count', 'comment_count', 'favorite_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Convert date columns
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        return df
    
    def _clean_text_data(self, df):
        """Clean text data (titles, descriptions, tags)"""
        logger.info("Cleaning text data...")
        
        text_cols = ['title', 'description', 'tags']
        
        for col in text_cols:
            if col in df.columns:
                # Convert to string
                df[col] = df[col].astype(str)
                
                # Remove URLs
                df[col] = df[col].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
                
                # Remove special characters but keep spaces
                df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s,]', ' ', x))
                
                # Remove extra whitespace
                df[col] = df[col].apply(lambda x: ' '.join(x.split()))
        
        return df
    
    def _create_features(self, df):
        """Create new features from existing data"""
        logger.info("Creating new features...")
        
        # Title features
        if 'title' in df.columns:
            df['title_length'] = df['title'].apply(len)
            df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))
            df['title_has_question'] = df['title'].apply(lambda x: 1 if '?' in x else 0)
            df['title_has_exclamation'] = df['title'].apply(lambda x: 1 if '!' in x else 0)
            df['title_uppercase_ratio'] = df['title'].apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            )
        
        # Description features
        if 'description' in df.columns:
            df['description_length'] = df['description'].apply(len)
            df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))
        
        # Tags features
        if 'tags' in df.columns:
            df['tag_count'] = df['tags'].apply(lambda x: len(x.split(',')) if x else 0)
            df['avg_tag_length'] = df['tags'].apply(
                lambda x: np.mean([len(tag) for tag in x.split(',')]) if x else 0
            )
        
        # Time features
        if 'published_at' in df.columns:
            df['publish_hour'] = df['published_at'].dt.hour
            df['publish_day_of_week'] = df['published_at'].dt.dayofweek
            df['publish_month'] = df['published_at'].dt.month
            df['publish_year'] = df['published_at'].dt.year
            df['is_weekend'] = df['publish_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Days since publish
            df['days_since_publish'] = (pd.Timestamp.now(tz=None) - df['published_at'].dt.tz_localize(None)).dt.days
        
        # Engagement features
        if all(col in df.columns for col in ['view_count', 'like_count', 'comment_count']):
            df['like_ratio'] = df['like_count'] / df['view_count'].replace(0, 1)
            df['comment_ratio'] = df['comment_count'] / df['view_count'].replace(0, 1)
            df['engagement_rate'] = (df['like_count'] + df['comment_count']) / df['view_count'].replace(0, 1)
        
        # Log-transformed view count (for better distribution)
        if 'view_count' in df.columns:
            df['log_view_count'] = np.log1p(df['view_count'])
        
        return df
    
    def prepare_for_regression(self, df):
        """Prepare features for regression model"""
        feature_cols = [
            'title_length', 'title_word_count', 'tag_count', 'avg_tag_length',
            'description_length', 'publish_hour', 'publish_day_of_week',
            'publish_month', 'is_weekend', 'like_ratio', 'comment_ratio'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].fillna(0)
        y = df['view_count'] if 'view_count' in df.columns else None
        
        return X, y
    
    def prepare_for_classification(self, df):
        """Prepare features for classification model"""
        # Combine text features
        text_features = []
        
        if 'title' in df.columns:
            text_features.append(df['title'])
        if 'description' in df.columns:
            text_features.append(df['description'])
        if 'tags' in df.columns:
            text_features.append(df['tags'])
        
        if text_features:
            X_text = text_features[0]
            for feat in text_features[1:]:
                X_text = X_text + ' ' + feat
        else:
            X_text = pd.Series([''] * len(df))
        
        y = df['category_name'] if 'category_name' in df.columns else None
        
        return X_text, y
    
    def get_processed_data(self, df):
        """Get fully processed data ready for analysis"""
        return self.preprocess(df)


def preprocess_data(df):
    """Main function to preprocess data"""
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(df)


if __name__ == "__main__":
    # Test preprocessing
    from data_collection import collect_data
    
    df = collect_data()
    processed_df = preprocess_data(df)
    print(f"Processed data shape: {processed_df.shape}")
    print(processed_df.columns.tolist())
