"""
Comprehensive Integration Tests for Sentiment Analysis Platform
Tests all components and their interactions
"""
import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = 'http://localhost:5000'
TEST_TEXTS = [
    "I absolutely love this amazing product! It's fantastic!",
    "This is terrible service. Very disappointed and frustrated.",
    "The product is okay, nothing special but it works fine.",
    "Outstanding experience! Highly recommend to everyone!",
    "Worst purchase ever. Complete waste of money and time."
]

def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

def print_test_result(test_name, success, details=None):
    """Print formatted test result."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{status} - {test_name}")
    if details:
        print(f"   Details: {details}")

def test_system_health():
    """Test system health endpoint."""
    print_test_header("System Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        data = response.json()
        
        success = (
            response.status_code == 200 and
            data.get('status') == 'healthy' and
            all(status == 'healthy' for status in data.get('components', {}).values())
        )
        
        print_test_result("System Health", success, f"Status: {data.get('status')}")
        
        if success:
            print("   Components:")
            for component, status in data.get('components', {}).items():
                print(f"     - {component}: {status}")
        
        return success
        
    except Exception as e:
        print_test_result("System Health", False, str(e))
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print_test_header("Sentiment Analysis Engine")
    
    all_passed = True
    
    for i, text in enumerate(TEST_TEXTS, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/api/sentiment/analyze",
                json={"text": text},
                timeout=15
            )
            
            data = response.json()
            success = (
                response.status_code == 200 and
                data.get('success') is True and
                'result' in data and
                'sentiment' in data['result'] and
                'confidence' in data['result']
            )
            
            if success:
                sentiment = data['result']['sentiment']
                confidence = data['result']['confidence']
                print_test_result(
                    f"Text {i} Analysis", 
                    True, 
                    f"Sentiment: {sentiment}, Confidence: {confidence:.3f}"
                )
            else:
                print_test_result(f"Text {i} Analysis", False, "Invalid response format")
                all_passed = False
                
        except Exception as e:
            print_test_result(f"Text {i} Analysis", False, str(e))
            all_passed = False
    
    return all_passed

def test_batch_analysis():
    """Test batch sentiment analysis."""
    print_test_header("Batch Analysis")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/sentiment/analyze",
            json={"texts": TEST_TEXTS[:3]},
            timeout=20
        )
        
        data = response.json()
        success = (
            response.status_code == 200 and
            data.get('success') is True and
            'results' in data and
            len(data['results']) == 3 and
            'batch_stats' in data
        )
        
        if success:
            batch_stats = data['batch_stats']
            print_test_result(
                "Batch Processing", 
                True, 
                f"Processed {batch_stats['total_texts']} texts, Avg confidence: {batch_stats['average_confidence']:.3f}"
            )
        else:
            print_test_result("Batch Processing", False, "Invalid batch response")
        
        return success
        
    except Exception as e:
        print_test_result("Batch Processing", False, str(e))
        return False

def test_streaming_system():
    """Test real-time streaming system."""
    print_test_header("Real-time Streaming System")
    
    all_passed = True
    
    # Test streaming start
    try:
        response = requests.post(
            f"{BASE_URL}/api/streaming/start",
            json={
                "tweets_per_minute": 60,
                "batch_size": 5,
                "processing_interval": 2.0
            },
            timeout=10
        )
        
        data = response.json()
        start_success = response.status_code == 200 and data.get('success') is True
        print_test_result("Streaming Start", start_success)
        
        if not start_success:
            all_passed = False
        
    except Exception as e:
        print_test_result("Streaming Start", False, str(e))
        all_passed = False
    
    # Wait for processing to begin
    time.sleep(5)
    
    # Test streaming status
    try:
        response = requests.get(f"{BASE_URL}/api/streaming/status", timeout=10)
        data = response.json()
        
        status_success = (
            response.status_code == 200 and
            data.get('success') is True and
            data.get('streaming_status', {}).get('stream_active') is True and
            data.get('streaming_status', {}).get('processing_active') is True
        )
        
        if status_success:
            status = data['streaming_status']
            print_test_result(
                "Streaming Status", 
                True, 
                f"Generated: {status['total_generated']}, Processed: {status['total_processed']}"
            )
        else:
            print_test_result("Streaming Status", False, "Stream not active")
            all_passed = False
        
    except Exception as e:
        print_test_result("Streaming Status", False, str(e))
        all_passed = False
    
    # Test recent tweets
    try:
        response = requests.get(f"{BASE_URL}/api/streaming/recent-tweets?count=5", timeout=10)
        data = response.json()
        
        tweets_success = (
            response.status_code == 200 and
            data.get('success') is True and
            len(data.get('tweets', [])) > 0
        )
        
        if tweets_success:
            tweet_count = len(data['tweets'])
            print_test_result("Recent Tweets", True, f"Retrieved {tweet_count} tweets")
        else:
            print_test_result("Recent Tweets", False, "No tweets retrieved")
            all_passed = False
        
    except Exception as e:
        print_test_result("Recent Tweets", False, str(e))
        all_passed = False
    
    # Test sentiment trends
    try:
        response = requests.get(f"{BASE_URL}/api/streaming/trends?time_window_minutes=10", timeout=10)
        data = response.json()
        
        trends_success = (
            response.status_code == 200 and
            data.get('success') is True and
            'trends' in data
        )
        
        if trends_success:
            trends = data['trends']
            print_test_result(
                "Sentiment Trends", 
                True, 
                f"Total tweets: {trends['total_tweets']}, Avg confidence: {trends['average_confidence']:.3f}"
            )
        else:
            print_test_result("Sentiment Trends", False, "Invalid trends data")
            all_passed = False
        
    except Exception as e:
        print_test_result("Sentiment Trends", False, str(e))
        all_passed = False
    
    return all_passed

def test_frontend_access():
    """Test frontend accessibility."""
    print_test_header("Frontend Dashboard")
    
    try:
        response = requests.get(BASE_URL, timeout=10)
        
        success = (
            response.status_code == 200 and
            'Sentiment Analysis Dashboard' in response.text and
            'Real-time Social Media Analytics' in response.text
        )
        
        print_test_result("Frontend Access", success, f"Status: {response.status_code}")
        return success
        
    except Exception as e:
        print_test_result("Frontend Access", False, str(e))
        return False

def test_error_handling():
    """Test error handling."""
    print_test_header("Error Handling")
    
    all_passed = True
    
    # Test invalid sentiment analysis request
    try:
        response = requests.post(
            f"{BASE_URL}/api/sentiment/analyze",
            json={"invalid_field": "test"},
            timeout=10
        )
        
        error_success = response.status_code == 400
        print_test_result("Invalid Request Handling", error_success)
        
        if not error_success:
            all_passed = False
        
    except Exception as e:
        print_test_result("Invalid Request Handling", False, str(e))
        all_passed = False
    
    # Test non-existent endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/nonexistent", timeout=10)
        not_found_success = response.status_code == 404
        print_test_result("404 Error Handling", not_found_success)
        
        if not not_found_success:
            all_passed = False
        
    except Exception as e:
        print_test_result("404 Error Handling", False, str(e))
        all_passed = False
    
    return all_passed

def run_performance_test():
    """Run basic performance test."""
    print_test_header("Performance Test")
    
    try:
        # Test multiple concurrent requests
        start_time = time.time()
        
        for i in range(10):
            response = requests.post(
                f"{BASE_URL}/api/sentiment/analyze",
                json={"text": f"Test message {i} for performance testing"},
                timeout=15
            )
            
            if response.status_code != 200:
                print_test_result("Performance Test", False, f"Request {i} failed")
                return False
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        success = avg_time < 2.0  # Average response time should be under 2 seconds
        print_test_result(
            "Performance Test", 
            success, 
            f"10 requests in {total_time:.2f}s (avg: {avg_time:.2f}s per request)"
        )
        
        return success
        
    except Exception as e:
        print_test_result("Performance Test", False, str(e))
        return False

def main():
    """Run all integration tests."""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🧪 Integration Test Suite 🧪                      ║
    ║                                                              ║
    ║     Comprehensive testing of Sentiment Analysis Platform    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Testing URL: {BASE_URL}
    Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    # Run all tests
    test_results = {
        "System Health": test_system_health(),
        "Sentiment Analysis": test_sentiment_analysis(),
        "Batch Analysis": test_batch_analysis(),
        "Streaming System": test_streaming_system(),
        "Frontend Access": test_frontend_access(),
        "Error Handling": test_error_handling(),
        "Performance": run_performance_test()
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall Result: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

