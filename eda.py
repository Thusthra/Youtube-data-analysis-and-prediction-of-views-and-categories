"""
Exploratory Data Analysis Module for YouTube Data Analysis
Handles data visualization and trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class YouTubeEDA:
    """Performs exploratory data analysis on YouTube video data"""
    
    def __init__(self, df):
        self.df = df
        self.output_dir = 'static/plots'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        logger.info("Generating all visualizations...")
        
        plots = {}
        
        # Distribution plots
        plots['view_distribution'] = self.plot_view_distribution()
        plots['category_distribution'] = self.plot_category_distribution()
        
        # Correlation plots
        plots['correlation_heatmap'] = self.plot_correlation_heatmap()
        
        # Time series plots
        plots['views_over_time'] = self.plot_views_over_time()
        plots['upload_patterns'] = self.plot_upload_patterns()
        
        # Engagement plots
        plots['engagement_analysis'] = self.plot_engagement_analysis()
        
        # Top videos
        plots['top_videos'] = self.plot_top_videos()
        
        logger.info(f"Generated {len(plots)} visualizations")
        
        return plots
    
    def plot_view_distribution(self):
        """Plot distribution of view counts"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=('View Count Distribution', 'Log View Count Distribution'))
        
        # Original distribution
        fig.add_trace(
            go.Histogram(x=self.df['view_count'], nbinsx=50, name='Views'),
            row=1, col=1
        )
        
        # Log distribution
        fig.add_trace(
            go.Histogram(x=self.df['log_view_count'], nbinsx=50, name='Log Views'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="View Count Distribution Analysis",
            showlegend=False,
            height=400
        )
        
        # Save as HTML
        fig.write_html(f'{self.output_dir}/view_distribution.html')
        
        # Also create matplotlib version
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['view_count'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('View Count')
        plt.ylabel('Frequency')
        plt.title('View Count Distribution')
        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        plt.subplot(1, 2, 2)
        plt.hist(self.df['log_view_count'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Log(View Count)')
        plt.ylabel('Frequency')
        plt.title('Log View Count Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/view_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'view_distribution'
    
    def plot_category_distribution(self):
        """Plot distribution of video categories"""
        if 'category_name' not in self.df.columns:
            logger.warning("Category name column not found")
            return None
        
        category_counts = self.df['category_name'].value_counts()
        
        # Plotly interactive chart
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={'x': 'Category', 'y': 'Number of Videos'},
            title='Video Distribution by Category'
        )
        
        fig.update_layout(xaxis_tickangle=-45, height=500)
        fig.write_html(f'{self.output_dir}/category_distribution.html')
        
        # Matplotlib version
        plt.figure(figsize=(12, 6))
        bars = plt.bar(category_counts.index, category_counts.values, edgecolor='black', alpha=0.8)
        plt.xlabel('Category')
        plt.ylabel('Number of Videos')
        plt.title('Video Distribution by Category')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/category_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'category_distribution'
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of numeric features"""
        numeric_cols = ['view_count', 'like_count', 'comment_count', 'title_length',
                       'tag_count', 'description_length', 'publish_hour', 'publish_day_of_week']
        
        # Filter to existing columns
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Plotly interactive heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=numeric_cols,
            y=numeric_cols,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Feature Correlation Heatmap'
        )
        
        fig.update_layout(height=600, width=700)
        fig.write_html(f'{self.output_dir}/correlation_heatmap.html')
        
        # Matplotlib version
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'correlation_heatmap'
    
    def plot_views_over_time(self):
        """Plot views over time"""
        if 'published_at' not in self.df.columns:
            logger.warning("Published date column not found")
            return None
        
        # Group by date
        daily_views = self.df.groupby(self.df['published_at'].dt.date).agg({
            'view_count': 'sum',
            'video_id': 'count'
        }).reset_index()
        
        daily_views.columns = ['date', 'total_views', 'video_count']
        
        # Plotly interactive chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Total Views Over Time', 'Videos Published Over Time'))
        
        fig.add_trace(
            go.Scatter(x=daily_views['date'], y=daily_views['total_views'],
                      mode='lines+markers', name='Total Views'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=daily_views['date'], y=daily_views['video_count'],
                  name='Video Count'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Views and Uploads Over Time")
        fig.write_html(f'{self.output_dir}/views_over_time.html')
        
        # Matplotlib version
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(daily_views['date'], daily_views['total_views'], marker='o', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Total Views')
        plt.title('Total Views Over Time')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        plt.bar(daily_views['date'], daily_views['video_count'], alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Number of Videos')
        plt.title('Videos Published Over Time')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/views_over_time.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'views_over_time'
    
    def plot_upload_patterns(self):
        """Plot upload patterns by hour and day"""
        if 'publish_hour' not in self.df.columns or 'publish_day_of_week' not in self.df.columns:
            logger.warning("Time feature columns not found")
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Uploads by Hour of Day', 'Uploads by Day of Week')
        )
        
        # Hour distribution
        hour_counts = self.df['publish_hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hour_counts.index, y=hour_counts.values, name='Hour'),
            row=1, col=1
        )
        
        # Day distribution
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.df['publish_day_of_week'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=day_names, y=day_counts.values, name='Day'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Upload Patterns", showlegend=False)
        fig.write_html(f'{self.output_dir}/upload_patterns.html')
        
        # Matplotlib version
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(hour_counts.index, hour_counts.values, edgecolor='black', alpha=0.8)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Videos')
        plt.title('Uploads by Hour of Day')
        plt.xticks(range(0, 24))
        
        plt.subplot(1, 2, 2)
        plt.bar(day_names, day_counts.values, edgecolor='black', alpha=0.8)
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Videos')
        plt.title('Uploads by Day of Week')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/upload_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'upload_patterns'
    
    def plot_engagement_analysis(self):
        """Plot engagement analysis"""
        if not all(col in self.df.columns for col in ['like_ratio', 'comment_ratio']):
            logger.warning("Engagement ratio columns not found")
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Like Ratio Distribution', 'Comment Ratio Distribution')
        )
        
        fig.add_trace(
            go.Histogram(x=self.df['like_ratio'], nbinsx=50, name='Like Ratio'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=self.df['comment_ratio'], nbinsx=50, name='Comment Ratio'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Engagement Analysis", showlegend=False)
        fig.write_html(f'{self.output_dir}/engagement_analysis.html')
        
        # Matplotlib version
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['like_ratio'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Like Ratio (Likes/Views)')
        plt.ylabel('Frequency')
        plt.title('Like Ratio Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.df['comment_ratio'], bins=50, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Comment Ratio (Comments/Views)')
        plt.ylabel('Frequency')
        plt.title('Comment Ratio Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/engagement_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'engagement_analysis'
    
    def plot_top_videos(self, n=20):
        """Plot top videos by views"""
        top_videos = self.df.nlargest(n, 'view_count')
        
        fig = px.bar(
            top_videos,
            x='view_count',
            y='title',
            orientation='h',
            title=f'Top {n} Videos by View Count',
            labels={'view_count': 'View Count', 'title': 'Video Title'}
        )
        
        fig.update_layout(height=600, yaxis={'autorange': 'reversed'})
        fig.write_html(f'{self.output_dir}/top_videos.html')
        
        # Matplotlib version
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_videos)), top_videos['view_count'], edgecolor='black', alpha=0.8)
        plt.yticks(range(len(top_videos)), [title[:50] + '...' if len(title) > 50 else title 
                                           for title in top_videos['title']])
        plt.xlabel('View Count')
        plt.title(f'Top {n} Videos by View Count')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_videos.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'top_videos'
    
    def get_summary_statistics(self):
        """Get summary statistics of the dataset"""
        stats = {
            'total_videos': len(self.df),
            'total_views': self.df['view_count'].sum(),
            'avg_views': self.df['view_count'].mean(),
            'median_views': self.df['view_count'].median(),
            'max_views': self.df['view_count'].max(),
            'min_views': self.df['view_count'].min(),
            'avg_likes': self.df['like_count'].mean() if 'like_count' in self.df.columns else 0,
            'avg_comments': self.df['comment_count'].mean() if 'comment_count' in self.df.columns else 0,
            'unique_categories': self.df['category_name'].nunique() if 'category_name' in self.df.columns else 0,
            'unique_channels': self.df['channel_title'].nunique() if 'channel_title' in self.df.columns else 0
        }
        
        return stats
    
    def get_category_stats(self):
        """Get statistics by category"""
        if 'category_name' not in self.df.columns:
            return {}
        
        category_stats = self.df.groupby('category_name').agg({
            'view_count': ['mean', 'median', 'sum', 'count'],
            'like_count': 'mean',
            'comment_count': 'mean'
        }).round(2)
        
        return category_stats.to_dict()


def perform_eda(df):
    """Main function to perform EDA"""
    eda = YouTubeEDA(df)
    plots = eda.generate_all_visualizations()
    stats = eda.get_summary_statistics()
    
    return plots, stats


if __name__ == "__main__":
    # Test EDA
    from data_collection import collect_data
    from data_preprocessing import preprocess_data
    
    df = collect_data()
    processed_df = preprocess_data(df)
    plots, stats = perform_eda(processed_df)
    
    print("Generated plots:", plots)
    print("\nSummary statistics:", stats)
