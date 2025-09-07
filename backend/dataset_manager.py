#!/usr/bin/env python3
"""
Dataset Management for Face Recognition Training
"""

import os
import sys
import subprocess
import requests
import zipfile
from pathlib import Path
import shutil

def setup_kaggle_api():
    """Setup Kaggle API for dataset download"""
    print("🔧 Setting up Kaggle API...")
    
    try:
        import kaggle
        print("✅ Kaggle API available")
        
        # Check if credentials are set
        try:
            kaggle.api.authenticate()
            print("✅ Kaggle credentials configured")
            return True
        except Exception as e:
            print(f"❌ Kaggle authentication failed: {e}")
            print("📋 To setup Kaggle API:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create new API token")
            print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\{username}\\.kaggle\\ (Windows)")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
            
    except ImportError:
        print("📦 Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return setup_kaggle_api()

def download_datasets_kaggle():
    """Download datasets using Kaggle API"""
    if not setup_kaggle_api():
        return False
        
    try:
        import kaggle
        
        print("📥 Downloading datasets from Kaggle...")
        
        # Download dataset 1
        if not os.path.exists('face-mask-lite-dataset.zip'):
            print("📥 Downloading face-mask-lite-dataset...")
            kaggle.api.dataset_download_files(
                'prasoonkottarathil/face-mask-lite-dataset', 
                path='.', 
                unzip=False
            )
            print("✅ Dataset 1 downloaded!")
            
        # Download dataset 2  
        if not os.path.exists('face-mask-dataset.zip'):
            print("📥 Downloading face-mask-dataset...")
            kaggle.api.dataset_download_files(
                'omkargurav/face-mask-dataset',
                path='.', 
                unzip=False
            )
            print("✅ Dataset 2 downloaded!")
            
        return True
        
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        return False

def extract_datasets():
    """Extract downloaded datasets"""
    print("📂 Extracting datasets...")
    
    try:
        # Extract dataset 1
        if os.path.exists('face-mask-lite-dataset.zip') and not os.path.exists('mask_dataset1'):
            print("📂 Extracting face-mask-lite-dataset...")
            with zipfile.ZipFile('face-mask-lite-dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('mask_dataset1')
            print("✅ Dataset 1 extracted!")
            
        # Extract dataset 2
        if os.path.exists('face-mask-dataset.zip') and not os.path.exists('mask_dataset2'):
            print("📂 Extracting face-mask-dataset...")
            with zipfile.ZipFile('face-mask-dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('mask_dataset2')
            print("✅ Dataset 2 extracted!")
            
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def create_sample_data():
    """Create sample data structure for testing"""
    print("📊 Creating sample data structure...")
    
    try:
        # Create directory structure
        dirs = [
            'mask_dataset1/with_mask',
            'mask_dataset1/without_mask',
            'mask_dataset2/train/with_mask',
            'mask_dataset2/train/without_mask',
            'mask_dataset2/test/with_mask', 
            'mask_dataset2/test/without_mask'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print("✅ Sample directory structure created!")
        print("📋 Directories created:")
        for dir_path in dirs:
            print(f"  - {dir_path}/")
            
        print("\n💡 To add real data:")
        print("1. Place face images in the created directories")
        print("2. Use .jpg, .png, .jpeg, .bmp formats")
        print("3. Ensure images contain clear faces")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def check_datasets():
    """Check available datasets"""
    print("🔍 Checking available datasets...")
    
    datasets = {
        'mask_dataset1': 0,
        'mask_dataset2': 0
    }
    
    for dataset_name in datasets.keys():
        if os.path.exists(dataset_name):
            # Count images recursively
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            count = 0
            
            for root, dirs, files in os.walk(dataset_name):
                count += sum(1 for file in files 
                           if any(file.lower().endswith(ext) for ext in image_extensions))
                           
            datasets[dataset_name] = count
            
    print("📊 Dataset Summary:")
    total_images = 0
    for dataset_name, count in datasets.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {dataset_name}: {count} images")
        total_images += count
        
    print(f"📈 Total images available: {total_images}")
    
    if total_images == 0:
        print("\n💡 No training data found. Options:")
        print("1. Run: python dataset_manager.py download")
        print("2. Run: python dataset_manager.py sample")
        print("3. Manually add images to mask_dataset1/ and mask_dataset2/")
        
    return total_images > 0

def clean_datasets():
    """Clean downloaded datasets"""
    print("🧹 Cleaning datasets...")
    
    items_to_remove = [
        'face-mask-lite-dataset.zip',
        'face-mask-dataset.zip', 
        'mask_dataset1',
        'mask_dataset2'
    ]
    
    for item in items_to_remove:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"🗑️  Removed directory: {item}")
            else:
                os.remove(item)
                print(f"🗑️  Removed file: {item}")
        else:
            print(f"⚪ Not found: {item}")
            
    print("✅ Cleanup complete!")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("📊 Face Recognition Dataset Manager")
        print("Usage:")
        print("  python dataset_manager.py download  - Download from Kaggle")
        print("  python dataset_manager.py extract   - Extract downloaded files")
        print("  python dataset_manager.py sample    - Create sample structure")
        print("  python dataset_manager.py check     - Check available data")
        print("  python dataset_manager.py clean     - Clean all datasets")
        print("  python dataset_manager.py setup     - Full setup (download + extract)")
        return
        
    command = sys.argv[1].lower()
    
    if command == "download":
        download_datasets_kaggle()
    elif command == "extract":
        extract_datasets()
    elif command == "sample":
        create_sample_data()
    elif command == "check":
        check_datasets()
    elif command == "clean":
        clean_datasets()
    elif command == "setup":
        if download_datasets_kaggle():
            extract_datasets()
        else:
            print("⚠️ Download failed, creating sample structure...")
            create_sample_data()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()