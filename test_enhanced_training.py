#!/usr/bin/env python3
"""
Test script for enhanced training system
Similar to your original notebook approach
"""
import requests
import json
import time
import base64
import numpy as np

API_BASE = "http://localhost:8000"

def test_enhanced_training():
    """Test the enhanced training system"""
    print("🧪 Testing Enhanced Training System")
    print("=" * 50)
    
    # 1. Check system status
    print("1. Checking system status...")
    try:
        response = requests.get(f"{API_BASE}/health")
        health = response.json()
        print(f"✅ System Status: {health['status']}")
        print(f"📊 Services: {health['services']}")
    except Exception as e:
        print(f"❌ System status check failed: {e}")
        return
    
    # 2. Check current users
    print("\n2. Checking registered users...")
    try:
        response = requests.get(f"{API_BASE}/api/faces/users")
        users = response.json()
        print(f"👥 Found {len(users)} registered users")
        for user in users:
            print(f"   - {user['name']}: {user.get('total_samples', 0)} samples")
    except Exception as e:
        print(f"❌ Users check failed: {e}")
        return
    
    if len(users) < 2:
        print("⚠️ Need at least 2 users for enhanced training")
        print("💡 Please register more users first")
        return
    
    # 3. Test enhanced training
    print("\n3. Starting enhanced training...")
    try:
        response = requests.post(f"{API_BASE}/api/faces/train-enhanced?use_synthetic=true")
        training_result = response.json()
        print(f"✅ Training initiated: {training_result['message']}")
        print(f"🔄 Status: {training_result['status']}")
        print(f"🧬 Synthetic data: {training_result['synthetic_data']}")
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        return
    
    # 4. Monitor training progress
    print("\n4. Monitoring training progress...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE}/api/faces/enhanced-training-status")
            status = response.json()
            
            is_training = status.get('is_training', False)
            stats = status.get('training_stats', {})
            
            if not is_training and stats:
                print(f"✅ Training completed!")
                print(f"🎯 Results:")
                print(f"   - Success: {stats.get('success', False)}")
                if stats.get('success'):
                    print(f"   - Best Model: {stats.get('best_model', 'Unknown')}")
                    print(f"   - Best Accuracy: {stats.get('best_accuracy', 0):.4f}")
                    print(f"   - SVM Accuracy: {stats.get('svm_accuracy', 0):.4f}")
                    print(f"   - RF Accuracy: {stats.get('rf_accuracy', 0):.4f}")
                    print(f"   - Total Samples: {stats.get('total_samples', 0)}")
                    print(f"   - Identities: {stats.get('num_identities', 0)}")
                    print(f"   - Enhanced Features: {stats.get('enhanced_features', False)}")
                else:
                    print(f"   - Message: {stats.get('message', 'Unknown error')}")
                break
            elif is_training:
                print(f"🔄 Training in progress... (attempt {attempt + 1}/{max_attempts})")
            else:
                print(f"⏳ Waiting for training to start... (attempt {attempt + 1}/{max_attempts})")
            
            time.sleep(2)
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            break
    else:
        print(f"⏰ Training monitoring timeout after {max_attempts * 2} seconds")
    
    # 5. Test basic training for comparison
    print("\n5. Testing basic training for comparison...")
    try:
        response = requests.post(f"{API_BASE}/api/faces/train-models")
        basic_result = response.json()
        print(f"✅ Basic training initiated: {basic_result['message']}")
    except Exception as e:
        print(f"❌ Basic training failed: {e}")
    
    # 6. Final system status
    print("\n6. Final system status...")
    try:
        response = requests.get(f"{API_BASE}/api/faces/system-status")
        final_status = response.json()
        print(f"🎯 System Ready: {final_status.get('system_ready', False)}")
        print(f"🤖 ML Model Trained: {final_status.get('ml_model_trained', False)}")
        print(f"👥 Total Users: {final_status.get('total_users', 0)}")
        print(f"📊 Face Database Size: {final_status.get('face_database_size', 0)}")
    except Exception as e:
        print(f"❌ Final status check failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced Training Test Complete!")
    print("\n💡 Next Steps:")
    print("1. Register more users with multiple photos each")
    print("2. Include photos with and without masks")
    print("3. Test live recognition to see improved accuracy")
    print("4. Use the Dashboard's Enhanced Training section")

if __name__ == "__main__":
    test_enhanced_training()