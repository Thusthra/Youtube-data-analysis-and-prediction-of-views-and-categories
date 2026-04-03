"""
Data Collection Module for YouTube Data Analysis
Handles fetching data from YouTube API or loading from CSV/JSON files
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDataCollector:
    """Collects YouTube video data from API or local files"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.youtube = None
        if api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=api_key)
                logger.info("YouTube API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API: {e}")
    
    def fetch_videos_by_search(self, query, max_results=50):
        """Fetch videos based on search query"""
        if not self.youtube:
            logger.warning("YouTube API not initialized. Using sample data.")
            return self._generate_sample_data()
        
        try:
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=max_results,
                type='video'
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                logger.warning("No videos found. Using sample data.")
                return self._generate_sample_data()
            
            return self._get_video_details(video_ids)
            
        except HttpError as e:
            logger.error(f"API Error: {e}")
            return self._generate_sample_data()
    
    def _get_video_details(self, video_ids):
        """Get detailed information for specific videos"""
        videos_data = []
        
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            try:
                video_response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch_ids)
                ).execute()
                
                for video in video_response.get('items', []):
                    snippet = video.get('snippet', {})
                    statistics = video.get('statistics', {})
                    
                    video_data = {
                        'video_id': video['id'],
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', ''),
                        'tags': ','.join(snippet.get('tags', [])),
                        'category_id': snippet.get('categoryId', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'channel_title': snippet.get('channelTitle', ''),
                        'view_count': int(statistics.get('viewCount', 0)),
                        'like_count': int(statistics.get('likeCount', 0)),
                        'comment_count': int(statistics.get('commentCount', 0)),
                        'favorite_count': int(statistics.get('favoriteCount', 0))
                    }
                    videos_data.append(video_data)
                    
            except HttpError as e:
                logger.error(f"Error fetching video details: {e}")
                continue
        
        return pd.DataFrame(videos_data)
    
    def load_from_csv(self, filepath):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return self._generate_sample_data()
    
    def load_from_json(self, filepath):
        """Load data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_samples=1000):
        """Generate sample YouTube data for demonstration"""
        logger.info(f"Generating {n_samples} sample records")
        
        np.random.seed(42)
        
        categories = [
            'Music', 'Entertainment', 'Education', 'Gaming', 'Sports',
            'News', 'Comedy', 'Film', 'Howto', 'Science', 'Technology',
            'Travel', 'Food', 'Fashion', 'Fitness', 'Music', 'Pets'
        ]
        
        titles = [
            "Amazing Tutorial You Must Watch",
            "Top 10 Tips for Success",
            "How to Master This Skill",
            "Epic Gaming Moments Compilation",
            "Breaking News Update",
            "Funny Moments That Will Make You Laugh",
            "Beautiful Travel Destination",
            "Delicious Recipe You Need to Try",
            "Workout Routine for Beginners",
            "Tech Review: Latest Gadgets",
            "Music Video - Official Release",
            "Educational Content Explained",
            "Sports Highlights of the Week",
            "Fashion Trends This Season",
            "Pet Compilation - Adorable Moments"
        ]
        
        descriptions = [
            "In this video, we explore the fascinating world of...",
            "Learn the best techniques and strategies...",
            "Watch as we dive deep into this topic...",
            "An incredible journey through...",
            "Step-by-step guide to achieving...",
            "The most comprehensive review of...",
            "Discover the secrets behind...",
            "Everything you need to know about...",
            "A complete breakdown of...",
            "The ultimate resource for..."
        ]
        
        tag_options = [
            "tutorial,howto,learn,education",
            "gaming,gameplay,esports,compilation",
            "news,breaking,update,current",
            "funny,comedy,humor,laugh",
            "travel,adventure,explore,destination",
            "food,recipe,cooking,delicious",
            "fitness,workout,exercise,health",
            "tech,review,gadget,technology",
            "music,song,official,video",
            "sports,highlights,game,match"
        ]
        
        data = []
        for i in range(n_samples):
            category = np.random.choice(categories)
            title = np.random.choice(titles)
            description = np.random.choice(descriptions)
            tags = np.random.choice(tag_options)
            
            # Generate realistic view counts (log-normal distribution)
            view_count = int(np.random.lognormal(mean=12, sigma=2))
            
            # Generate correlated metrics
            like_ratio = np.random.uniform(0.01, 0.15)
            comment_ratio = np.random.uniform(0.001, 0.01)
            
            like_count = int(view_count * like_ratio)
            comment_count = int(view_count * comment_ratio)
            
            # Generate publish date within last 2 years
            days_ago = np.random.randint(0, 730)
            publish_date = datetime.now() - pd.Timedelta(days=days_ago)
            
            data.append({
                'video_id': f'video_{i:04d}',
                'title': title,
                'description': description,
                'tags': tags,
                'category_id': str(hash(category) % 20 + 1),
                'category_name': category,
                'published_at': publish_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'channel_title': f'Channel_{np.random.randint(1, 50)}',
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'favorite_count': np.random.randint(0, 100)
            })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/youtube_sample_data.csv', index=False)
        logger.info("Sample data saved to data/youtube_sample_data.csv")
        
        return df
    
    def get_category_mapping(self):
        """Get YouTube category ID to name mapping"""
        return {
            '1': 'Film & Animation',
            '2': 'Autos & Vehicles',
            '10': 'Music',
            '15': 'Pets & Animals',
            '17': 'Sports',
            '18': 'Short Movies',
            '19': 'Travel & Events',
            '20': 'Gaming',
            '21': 'Videoblogging',
            '22': 'People & Blogs',
            '23': 'Comedy',
            '24': 'Entertainment',
            '25': 'News & Politics',
            '26': 'Howto & Style',
            '27': 'Education',
            '28': 'Science & Technology',
            '29': 'Nonprofits & Activism'
        }


def collect_data(api_key=None, query="trending", max_results=100, filepath=None):
    """Main function to collect YouTube data"""
    collector = YouTubeDataCollector(api_key)
    
    if filepath:
        if filepath.endswith('.csv'):
            return collector.load_from_csv(filepath)
        elif filepath.endswith('.json'):
            return collector.load_from_json(filepath)
    
    return collector.fetch_videos_by_search(query, max_results)


if __name__ == "__main__":
    # Test data collection
    df = collect_data()
    print(f"Collected {len(df)} videos")
    print(df.head())
