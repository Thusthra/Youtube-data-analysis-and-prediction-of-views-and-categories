"""
Flask Web Application for YouTube Data Analysis and Prediction
Provides interactive user interface for predictions and visualizations
"""

# Fix pyparsing compatibility issue
import pyparsing
if not hasattr(pyparsing, 'DelimitedList'):
    pyparsing.DelimitedList = pyparsing.delimitedList

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from data_collection import YouTubeDataCollector, collect_data
from data_preprocessing import DataPreprocessor, preprocess_data
from eda import YouTubeEDA, perform_eda
from view_prediction import ViewPredictor, train_view_prediction
from category_prediction import CategoryPredictor, train_category_prediction

app = Flask(__name__)

# Global variables to store models and data
data = None
processed_data = None
view_predictor = None
category_predictor = None
eda_plots = None
eda_stats = None


def initialize_system():
    """Initialize the system with data and models"""
    global data, processed_data, view_predictor, category_predictor, eda_plots, eda_stats
    
    print("Initializing YouTube Analysis System...")
    
    # Load or generate data
    if os.path.exists('data/youtube_sample_data.csv'):
        print("Loading existing data...")
        data = pd.read_csv('data/youtube_sample_data.csv')
    else:
        print("Generating sample data...")
        collector = YouTubeDataCollector()
        data = collector._generate_sample_data(1000)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_data(data)
    
    # Train view prediction model
    print("Training view prediction model...")
    view_predictor, view_results = train_view_prediction(processed_data)
    
    # Train category prediction model
    print("Training category prediction model...")
    category_predictor, category_results = train_category_prediction(processed_data)
    
    # Generate EDA visualizations
    print("Generating visualizations...")
    eda = YouTubeEDA(processed_data)
    eda_plots = eda.generate_all_visualizations()
    eda_stats = eda.get_summary_statistics()
    
    print("System initialized successfully!")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page with visualizations"""
    return render_template('dashboard.html', 
                         plots=eda_plots, 
                         stats=eda_stats)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        # Get form data
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        tags = request.form.get('tags', '')
        
        # Make predictions
        view_prediction = view_predictor.predict_from_text(title, description, tags)
        category_prediction = category_predictor.predict_from_video_info(title, description, tags)
        
        return render_template('predict.html',
                             title=title,
                             description=description,
                             tags=tags,
                             view_prediction=view_prediction,
                             category_prediction=category_prediction,
                             show_results=True)
    
    return render_template('predict.html', show_results=False)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        title = data.get('title', '')
        description = data.get('description', '')
        tags = data.get('tags', '')
        
        # Make predictions
        view_prediction = view_predictor.predict_from_text(title, description, tags)
        category_prediction = category_predictor.predict_from_video_info(title, description, tags)
        
        return jsonify({
            'success': True,
            'predictions': {
                'views': int(view_prediction),
                'category': category_prediction['category'],
                'confidence': float(category_prediction['confidence']) if category_prediction['confidence'] else None
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/analysis')
def analysis():
    """Analysis page with detailed statistics"""
    category_stats = eda.get_category_stats() if 'eda' in globals() else {}
    
    return render_template('analysis.html',
                         stats=eda_stats,
                         category_stats=category_stats)


@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify({
        'success': True,
        'stats': eda_stats,
        'category_stats': eda.get_category_stats() if 'eda' in globals() else {}
    })


@app.route('/api/data')
def api_data():
    """API endpoint for sample data"""
    try:
        # Return first 100 rows of processed data
        sample_data = processed_data.head(100).to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'data': sample_data,
            'total_records': len(processed_data)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/compare')
def compare():
    """Compare multiple videos"""
    return render_template('compare.html')


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """API endpoint for comparing videos"""
    try:
        data = request.get_json()
        videos = data.get('videos', [])
        
        results = []
        for video in videos:
            title = video.get('title', '')
            description = video.get('description', '')
            tags = video.get('tags', '')
            
            view_pred = view_predictor.predict_from_text(title, description, tags)
            cat_pred = category_predictor.predict_from_video_info(title, description, tags)
            
            results.append({
                'title': title,
                'predicted_views': int(view_pred),
                'predicted_category': cat_pred['category'],
                'confidence': float(cat_pred['confidence']) if cat_pred['confidence'] else None
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
