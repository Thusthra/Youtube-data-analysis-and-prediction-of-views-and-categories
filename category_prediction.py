"""
Category Prediction Module for YouTube Data Analysis
Implements classification models to predict video categories
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryPredictor:
    """Predicts YouTube video categories using classification models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """Prepare features for classification"""
        # Combine text features
        text_features = []
        
        if 'title' in df.columns:
            text_features.append(df['title'].fillna(''))
        if 'description' in df.columns:
            text_features.append(df['description'].fillna(''))
        if 'tags' in df.columns:
            text_features.append(df['tags'].fillna(''))
        
        if not text_features:
            logger.error("No text features available for classification")
            return None, None
        
        # Combine all text features
        X_text = text_features[0]
        for feat in text_features[1:]:
            X_text = X_text + ' ' + feat
        
        # Get target variable
        if 'category_name' not in df.columns:
            logger.error("Category name column not found")
            return None, None
        
        y = df['category_name']
        
        return X_text, y
    
    def train_models(self, df, test_size=0.2):
        """Train multiple classification models"""
        logger.info("Preparing features for classification...")
        
        X_text, y = self.prepare_features(df)
        
        if X_text is None or y is None:
            logger.error("Failed to prepare features")
            return None
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Linear SVC': LinearSVC(
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
            
            # Get classification report
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'classification_report': report
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model based on accuracy
        self.best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        self.models = results
        
        logger.info(f"Best model: {self.best_model_name} with accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        
        return {
            'results': results,
            'best_model': self.best_model_name,
            'X_test': X_test_vec,
            'y_test': y_test,
            'label_encoder': self.label_encoder
        }
    
    def predict(self, text):
        """Predict category for new text"""
        if self.best_model is None:
            logger.error("No model trained. Please train models first.")
            return None
        
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.best_model.predict(text_vec)
        
        # Decode label
        category = self.label_encoder.inverse_transform(prediction)[0]
        
        # Get probability if available
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(text_vec)[0]
            confidence = max(probabilities)
        else:
            confidence = None
        
        return {
            'category': category,
            'confidence': confidence
        }
    
    def predict_from_video_info(self, title, description='', tags=''):
        """Predict category from video information"""
        # Combine text features
        text = title
        if description:
            text += ' ' + description
        if tags:
            text += ' ' + tags
        
        return self.predict(text)
    
    def get_top_features(self, n=20):
        """Get top features for each category"""
        if self.best_model_name not in ['Logistic Regression', 'Linear SVC']:
            logger.warning("Feature importance only available for linear models")
            return None
        
        model = self.best_model
        feature_names = self.vectorizer.get_feature_names_out()
        
        top_features = {}
        
        for i, category in enumerate(self.label_encoder.classes_):
            if hasattr(model, 'coef_'):
                # For multi-class, get coefficients for this class
                if len(model.coef_.shape) > 1:
                    coef = model.coef_[i]
                else:
                    coef = model.coef_[0]
                
                # Get top positive and negative features
                top_positive_idx = coef.argsort()[-n:][::-1]
                top_negative_idx = coef.argsort()[:n]
                
                top_features[category] = {
                    'positive': [(feature_names[idx], coef[idx]) for idx in top_positive_idx],
                    'negative': [(feature_names[idx], coef[idx]) for idx in top_negative_idx]
                }
        
        return top_features
    
    def get_confusion_matrix(self, y_test, y_pred):
        """Get confusion matrix"""
        return confusion_matrix(y_test, y_pred)
    
    def save_models(self):
        """Save trained models to disk"""
        for name, result in self.models.items():
            filename = f"{self.model_dir}/{name.replace(' ', '_').lower()}_classifier.pkl"
            joblib.dump(result['model'], filename)
            logger.info(f"Saved {name} to {filename}")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, f"{self.model_dir}/tfidf_vectorizer.pkl")
        
        # Save label encoder
        joblib.dump(self.label_encoder, f"{self.model_dir}/label_encoder.pkl")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load vectorizer
            self.vectorizer = joblib.load(f"{self.model_dir}/tfidf_vectorizer.pkl")
            
            # Load label encoder
            self.label_encoder = joblib.load(f"{self.model_dir}/label_encoder.pkl")
            
            # Load models
            model_files = {
                'Logistic Regression': 'logistic_regression_classifier.pkl',
                'Naive Bayes': 'naive_bayes_classifier.pkl',
                'Linear SVC': 'linear_svc_classifier.pkl',
                'Random Forest': 'random_forest_classifier.pkl'
            }
            
            for name, filename in model_files.items():
                filepath = f"{self.model_dir}/{filename}"
                if os.path.exists(filepath):
                    self.models[name] = {'model': joblib.load(filepath)}
                    logger.info(f"Loaded {name} from {filepath}")
            
            # Set best model (prefer Logistic Regression)
            if 'Logistic Regression' in self.models:
                self.best_model_name = 'Logistic Regression'
                self.best_model = self.models['Logistic Regression']['model']
            elif self.models:
                self.best_model_name = list(self.models.keys())[0]
                self.best_model = list(self.models.values())[0]['model']
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.best_model is None:
            return None
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics


def train_category_prediction(df):
    """Main function to train category prediction models"""
    predictor = CategoryPredictor()
    results = predictor.train_models(df)
    
    if results:
        predictor.save_models()
    
    return predictor, results


if __name__ == "__main__":
    # Test category prediction
    from data_collection import collect_data
    from data_preprocessing import preprocess_data
    
    df = collect_data()
    processed_df = preprocess_data(df)
    
    predictor, results = train_category_prediction(processed_df)
    
    if results:
        print(f"\nBest model: {results['best_model']}")
        print("\nModel Performance:")
        for name, result in results['results'].items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        # Test prediction
        test_prediction = predictor.predict_from_video_info(
            title="How to Play Guitar - Beginner Tutorial",
            description="Learn guitar basics in this comprehensive tutorial",
            tags="guitar,tutorial,music,learn,howto"
        )
        print(f"\nPredicted category: {test_prediction['category']}")
        if test_prediction['confidence']:
            print(f"Confidence: {test_prediction['confidence']:.2%}")
