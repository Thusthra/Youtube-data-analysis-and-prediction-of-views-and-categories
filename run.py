#!/usr/bin/env python
"""
YouTube Data Analysis - Run Script
Easy startup script for the application
"""

# Fix pyparsing compatibility issue
import pyparsing
if not hasattr(pyparsing, 'DelimitedList'):
    pyparsing.DelimitedList = pyparsing.delimitedList

import os
import sys
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} not installed")
    
    if missing_packages:
        print("\nMissing packages detected!")
        print("Installing required packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✓ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("Error: Failed to install packages")
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'static/plots']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    import nltk
    
    nltk_packages = ['punkt', 'stopwords', 'wordnet']
    
    for package in nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            print(f"✓ NLTK {package} already downloaded")
        except LookupError:
            print(f"Downloading NLTK {package}...")
            nltk.download(package, quiet=True)
            print(f"✓ NLTK {package} downloaded")

def main():
    """Main function to run the application"""
    print("=" * 60)
    print("YouTube Data Analysis and Prediction System")
    print("=" * 60)
    print()
    
    # Check Python version
    check_python_version()
    print()
    
    # Check and install dependencies
    check_dependencies()
    print()
    
    # Create directories
    create_directories()
    print()
    
    # Download NLTK data
    download_nltk_data()
    print()
    
    # Start the application
    print("=" * 60)
    print("Starting Flask application...")
    print("=" * 60)
    print()
    print("The application will be available at:")
    print("  http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Give a moment for the user to read
    time.sleep(2)
    
    # Import and run the Flask app
    try:
        from app import app, initialize_system
        
        # Initialize the system
        print("Initializing system...")
        initialize_system()
        print()
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        print("Thank you for using YouTube Data Analysis!")
    except Exception as e:
        print(f"\nError starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if port 5000 is available")
        print("3. Verify Python version is 3.8 or higher")
        sys.exit(1)

if __name__ == '__main__':
    main()
