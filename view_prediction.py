"""
View Prediction Module for YouTube Data Analysis
Implements regression models to predict video view counts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViewPredictor:
    """Predicts YouTube video view counts using regression models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_model = None
        self.best_model_name = None
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """Prepare features for regression"""
        feature_cols = [
            'title_length', 'title_word_count', 'tag_count', 'avg_tag_length',
            'description_length', 'publish_hour', 'publish_day_of_week',
            'publish_month', 'is_weekend', 'like_ratio', 'comment_ratio'
        ]
        
        # Filter to existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            logger.error("No feature columns available for regression")
            return None, None
        
        self.feature_columns = available_cols
        
        X = df[available_cols].fillna(0)
        y = df['view_count'] if 'view_count' in df.columns else None
        
        return X, y
    
    def train_models(self, df, test_size=0.2):
        """Train multiple regression models"""
        logger.info("Preparing features for regression...")
        
        X, y = self.prepare_features(df)
        
        if X is None or y is None:
            logger.error("Failed to prepare features")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Use scaled data for Linear Regression, original for tree-based
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if name == 'Linear Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            logger.info(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model based on R² score
        self.best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[self.best_model_name]['model']
        self.models = results
        
        logger.info(f"Best model: {self.best_model_name} with R²: {results[self.best_model_name]['r2']:.4f}")
        
        return {
            'results': results,
            'best_model': self.best_model_name,
            'X_test': X_test,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }
    
    def predict(self, features):
        """Predict view count for new data"""
        if self.best_model is None:
            logger.error("No model trained. Please train models first.")
            return None
        
        # Ensure features are in correct format
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Select only feature columns
        features = features[self.feature_columns].fillna(0)
        
        # Scale if using Linear Regression
        if self.best_model_name == 'Linear Regression':
            features = self.scaler.transform(features)
        
        prediction = self.best_model.predict(features)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def predict_from_text(self, title, description='', tags=''):
        """Predict views based on text features"""
        # Extract features from text
        features = {
            'title_length': len(title),
            'title_word_count': len(title.split()),
            'tag_count': len(tags.split(',')) if tags else 0,
            'avg_tag_length': np.mean([len(tag) for tag in tags.split(',')]) if tags else 0,
            'description_length': len(description),
            'publish_hour': 12,  # Default to noon
            'publish_day_of_week': 3,  # Default to Wednesday
            'publish_month': 6,  # Default to June
            'is_weekend': 0,
            'like_ratio': 0.05,  # Default engagement ratios
            'comment_ratio': 0.005
        }
        
        return self.predict(features)
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        importance_dict = {}
        
        for name, result in self.models.items():
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_dict[name] = dict(zip(self.feature_columns, importance))
        
        return importance_dict
    
    def save_models(self):
        """Save trained models to disk"""
        for name, result in self.models.items():
            filename = f"{self.model_dir}/{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(result['model'], filename)
            logger.info(f"Saved {name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{self.model_dir}/feature_columns.pkl")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load scaler
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            
            # Load feature columns
            self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
            
            # Load models
            model_files = {
                'Linear Regression': 'linear_regression_model.pkl',
                'Random Forest': 'random_forest_model.pkl',
                'Gradient Boosting': 'gradient_boosting_model.pkl'
            }
            
            for name, filename in model_files.items():
                filepath = f"{self.model_dir}/{filename}"
                if os.path.exists(filepath):
                    self.models[name] = {'model': joblib.load(filepath)}
                    logger.info(f"Loaded {name} from {filepath}")
            
            # Set best model (prefer Random Forest)
            if 'Random Forest' in self.models:
                self.best_model_name = 'Random Forest'
                self.best_model = self.models['Random Forest']['model']
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
        
        # Scale if needed
        if self.best_model_name == 'Linear Regression':
            X_test = self.scaler.transform(X_test)
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics


def train_view_prediction(df):
    """Main function to train view prediction models"""
    predictor = ViewPredictor()
    results = predictor.train_models(df)
    
    if results:
        predictor.save_models()
    
    return predictor, results


if __name__ == "__main__":
    # Test view prediction
    from data_collection import collect_data
    from data_preprocessing import preprocess_data
    
    df = collect_data()
    processed_df = preprocess_data(df)
    
    predictor, results = train_view_prediction(processed_df)
    
    if results:
        print(f"\nBest model: {results['best_model']}")
        print("\nModel Performance:")
        for name, result in results['results'].items():
            print(f"\n{name}:")
            print(f"  RMSE: {result['rmse']:.2f}")
            print(f"  MAE: {result['mae']:.2f}")
            print(f"  R²: {result['r2']:.4f}")
            print(f"  CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        # Test prediction
        test_prediction = predictor.predict_from_text(
            title="Amazing Tutorial Video",
            description="Learn how to code in Python",
            tags="python,tutorial,programming"
        )
        print(f"\nPredicted views for test video: {test_prediction:,.0f}")
