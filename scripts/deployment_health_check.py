#!/usr/bin/env python3
"""
Post-deployment health check script.
Verify production deployment is working correctly.
"""
import sys
import time
import requests
import json
from pathlib import Path

def test_model_endpoint(base_url="http://localhost:3000"):
    """Test the model prediction endpoint."""
    print("🌐 Testing Model Endpoint...")
    
    test_cases = [
        {
            "message": "People trapped on roof. Send search and rescue.",
            "expected_labels": ["search_and_rescue"],
            "description": "Search and rescue scenario"
        },
        {
            "message": "Storm destroyed houses",
            "expected_labels": ["weather_related"],
            "description": "Weather-related disaster"
        },
        {
            "message": "Hospital damaged in earthquake",
            "expected_labels": ["hospitals"],
            "description": "Hospital infrastructure damage"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Make request
            start_time = time.time()
            response = requests.post(
                f"{base_url}/classify",
                json={"message": test_case["message"]},
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Test {i} ({test_case['description']}): {response_time:.0f}ms")
                
                # Check response time
                if response_time > 500:
                    print(f"⚠️ Response time {response_time:.0f}ms exceeds 500ms threshold")
                    all_passed = False
                
                # Check expected labels are present (basic validation)
                if 'predictions' in result:
                    predictions = result['predictions']
                    for expected_label in test_case['expected_labels']:
                        if expected_label in predictions:
                            print(f"   {expected_label}: {predictions[expected_label]}")
                        else:
                            print(f"   ⚠️ Expected label {expected_label} not found")
                
            else:
                print(f"❌ Test {i}: HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Test {i}: Request failed - {e}")
            all_passed = False
        except Exception as e:
            print(f"❌ Test {i}: Unexpected error - {e}")
            all_passed = False
    
    return all_passed

def test_health_endpoint(base_url="http://localhost:3000"):
    """Test the health check endpoint."""
    print("\n💓 Testing Health Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Health endpoint responding")
            print(f"   Status: {result.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Health endpoint returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_load_performance(base_url="http://localhost:3000", num_requests=10):
    """Test performance under load."""
    print(f"\n⚡ Testing Load Performance ({num_requests} requests)...")
    
    test_message = "Emergency situation requires immediate response"
    response_times = []
    errors = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/classify",
                json={"message": test_message},
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                response_times.append(response_time)
            else:
                errors += 1
                
        except Exception as e:
            errors += 1
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"✅ Load test completed:")
        print(f"   Average response time: {avg_time:.0f}ms")
        print(f"   Min/Max: {min_time:.0f}ms / {max_time:.0f}ms")
        print(f"   Success rate: {(num_requests-errors)/num_requests*100:.1f}%")
        
        # Check thresholds
        if avg_time > 500:
            print(f"⚠️ Average response time {avg_time:.0f}ms exceeds 500ms threshold")
            return False
        if errors > 0:
            print(f"⚠️ {errors} errors occurred during load test")
            return False
        
        return True
    else:
        print("❌ No successful requests during load test")
        return False

def main():
    """Run all health checks."""
    print("🏥 Post-Deployment Health Check")
    print("=" * 50)
    
    # You can modify this URL based on your deployment
    base_url = "http://localhost:3000"
    
    print(f"Testing deployment at: {base_url}")
    
    tests = [
        ("Health Endpoint", lambda: test_health_endpoint(base_url)),
        ("Model Endpoint", lambda: test_model_endpoint(base_url)),
        ("Load Performance", lambda: test_load_performance(base_url, 5))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ HEALTHY" if result else "❌ UNHEALTHY"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 DEPLOYMENT IS HEALTHY - PRODUCTION READY!")
        return 0
    else:
        print("⚠️ DEPLOYMENT ISSUES DETECTED - INVESTIGATE")
        return 1

if __name__ == '__main__':
    sys.exit(main())
