#!/usr/bin/env python3
"""
Simple test to download datasets without threading
"""

import os
import sys

def test_kaggle_download():
    """Test basic Kaggle download"""
    try:
        # Import kaggle
        import kaggle
        print("✅ Kaggle API available")
        
        # Setup credentials  
        os.environ['KAGGLE_USERNAME'] = 'soundaryats24cse'
        os.environ['KAGGLE_KEY'] = 'ed8d7faa7b233596c35babba8761c00b'
        
        # Authenticate
        kaggle.api.authenticate()
        print("✅ Kaggle authentication successful!")
        
        # Download tiny dataset (0.057 MB)
        print("📥 Downloading tiny dataset...")
        kaggle.api.dataset_download_files(
            'dataturks/face-detection-in-images', 
            path='.', 
            unzip=False
        )
        
        if os.path.exists('face-detection-in-images.zip'):
            size = os.path.getsize('face-detection-in-images.zip') / 1024
            print(f"✅ Downloaded! ({size:.1f} KB)")
        else:
            print("❌ Download failed - file not found")
            
        # Download main dataset (163 MB)
        print("📥 Downloading main dataset...")
        kaggle.api.dataset_download_files(
            'omkargurav/face-mask-dataset',
            path='.', 
            unzip=False
        )
        
        if os.path.exists('face-mask-dataset.zip'):
            size = os.path.getsize('face-mask-dataset.zip') / (1024*1024)
            print(f"✅ Downloaded! ({size:.1f} MB)")
        else:
            print("❌ Download failed - file not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kaggle_download()