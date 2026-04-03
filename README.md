# YouTube Data Analysis and Prediction System

A comprehensive machine learning system for analyzing YouTube video data and predicting views and categories.

## Features

### 1. Data Collection Module
- Fetch data using YouTube Data API
- Load datasets from CSV/JSON files
- Extract video titles, views, likes, comments, categories, and tags

### 2. Data Preprocessing Module
- Handle missing/null values
- Remove duplicates
- Convert data types (dates, numeric)
- Text cleaning (stopword removal, tokenization)
- Feature creation (title length, tag count, upload time features)

### 3. Exploratory Data Analysis Module
- Visualize trends using Matplotlib, Seaborn, and Plotly
- Identify most popular categories
- Analyze peak upload times
- Correlation analysis between likes, comments, and views
- Generate interactive charts and graphs

### 4. View Prediction Module (Regression)
- Train models using Scikit-learn
- Algorithms: Linear Regression, Random Forest, Gradient Boosting
- Predict expected views
- Evaluate performance: RMSE, R² score, MAE

### 5. Category Prediction Module (Classification)
- Apply NLP using TF-IDF
- Train classifiers: Logistic Regression, Naive Bayes, SVM, Random Forest
- Predict category based on title, description, and tags
- Measure accuracy, precision, recall

### 6. Visualization & Dashboard Module
- Interactive charts and dashboards
- Display predictions, trends, and category distribution
- Built with Plotly and Chart.js

### 7. Deployment Module
- Web application built with Flask
- User-friendly interface for predictions
- Video comparison feature
- Detailed analysis pages

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or download the project:
```bash
cd youtube_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up YouTube Data API:
   - Get an API key from Google Cloud Console
   - Create a `.env` file and add: `YOUTUBE_API_KEY=your_api_key_here`

## Usage

### Running the Application

1. Start the Flask web server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

### Using the Web Interface

#### Home Page
- Quick prediction form for instant results
- Overview of system features
- Statistics about the dataset

#### Dashboard
- Interactive visualizations of YouTube data
- Category distribution charts
- View count analysis
- Upload pattern analysis
- Engagement metrics

#### Predict Page
- Enter video title, description, and tags
- Get predicted view count
- Get suggested category with confidence score
- Receive optimization recommendations

#### Compare Page
- Compare multiple videos side by side
- Visual comparison charts
- Identify best performing video

#### Analysis Page
- Detailed statistics and insights
- Category-wise analysis
- Recommendations for content creators

### API Endpoints

#### Predict Views and Category
```bash
POST /api/predict
Content-Type: application/json

{
    "title": "Your Video Title",
    "description": "Video description",
    "tags": "tag1,tag2,tag3"
}
```

Response:
```json
{
    "success": true,
    "predictions": {
        "views": 125000,
        "category": "Education",
        "confidence": 0.85
    }
}
```

#### Compare Videos
```bash
POST /api/compare
Content-Type: application/json

{
    "videos": [
        {
            "title": "Video 1 Title",
            "description": "Description 1",
            "tags": "tag1,tag2"
        },
        {
            "title": "Video 2 Title",
            "description": "Description 2",
            "tags": "tag3,tag4"
        }
    ]
}
```

#### Get Statistics
```bash
GET /api/stats
```

#### Get Sample Data
```bash
GET /api/data
```

## Project Structure

```
youtube_analysis/
├── app.py                    # Flask web application
├── data_collection.py        # Data collection module
├── data_preprocessing.py     # Data preprocessing module
├── eda.py                    # Exploratory data analysis
├── view_prediction.py        # View prediction (regression)
├── category_prediction.py    # Category prediction (classification)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   └── youtube_sample_data.csv
├── models/                   # Trained models
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── logistic_regression_classifier.pkl
│   ├── naive_bayes_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── static/                   # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── plots/                # Generated visualizations
└── templates/                # HTML templates
    ├── base.html
    ├── index.html
    ├── predict.html
    ├── dashboard.html
    ├── compare.html
    ├── analysis.html
    └── about.html
```

## Machine Learning Models

### View Prediction (Regression)
- **Linear Regression**: Simple linear model for baseline predictions
- **Random Forest**: Ensemble method for better accuracy
- **Gradient Boosting**: Advanced ensemble for optimal performance

### Category Prediction (Classification)
- **Logistic Regression**: Fast and interpretable classifier
- **Naive Bayes**: Probabilistic classifier for text data
- **Linear SVM**: Support vector machine for high-dimensional data
- **Random Forest**: Ensemble classifier for robust predictions

## Features Used for Prediction

### View Prediction Features
- Title length
- Title word count
- Tag count
- Average tag length
- Description length
- Publish hour
- Publish day of week
- Publish month
- Is weekend
- Like ratio
- Comment ratio

### Category Prediction Features
- Title text (TF-IDF)
- Description text (TF-IDF)
- Tags text (TF-IDF)

## Technologies Used

- **Python 3.8+**: Core programming language
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **TensorFlow**: Deep learning (optional)
- **NLTK**: Natural language processing
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical graphics
- **Plotly**: Interactive charts
- **Bootstrap 5**: UI framework
- **Chart.js**: Client-side charts

## Sample Data

The system includes a sample data generator that creates realistic YouTube video data for demonstration purposes. The sample data includes:
- 1000 video records
- 17 different categories
- Realistic view count distributions
- Correlated engagement metrics

## Performance Metrics

### View Prediction
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Cross-validation score

### Category Prediction
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Contributing

Feel free to contribute to this project by:
1. Adding new features
2. Improving model accuracy
3. Enhancing the UI/UX
4. Adding more visualizations
5. Optimizing performance

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- YouTube Data API for data access
- Scikit-learn for machine learning algorithms
- Flask for web framework
- Bootstrap for UI components
- Plotly for interactive visualizations

## Contact

For questions or suggestions, please open an issue in the project repository.
