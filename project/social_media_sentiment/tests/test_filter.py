"""
Test Script for Relevance Filter
Tests the filtering policy with various example posts to demonstrate how it works.
"""

from ..relevance_filter import RelevanceFilter
import json


def test_relevance_filter():
    """Test the relevance filter with example posts."""
    
    print("="*80)
    print("RELEVANCE FILTER TEST")
    print("="*80)
    print("\nTesting the filtering policy with various example posts...\n")
    
    # Initialize filter
    filter_system = RelevanceFilter(
        min_relevance_score=1,
        min_quality_score=40,
        min_account_score=20
    )
    
    # Test cases
    test_posts = [
        {
            'name': 'RELEVANT: Strong Trading Intent',
            'text': '$AAPL looks great here. Buying 100 shares at $175. Target price $200, stop loss at $170.',
            'account_age_days': 365,
            'karma': 5000,
            'engagement': 50,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'stocks',
            'expected': True
        },
        {
            'name': 'RELEVANT: Technical Analysis',
            'text': 'TSLA breaking resistance at $250. RSI showing overbought but momentum is strong. Could see a rally to $280 if volume holds.',
            'account_age_days': 730,
            'karma': 15000,
            'engagement': 75,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'wallstreetbets',
            'expected': True
        },
        {
            'name': 'RELEVANT: Due Diligence',
            'text': 'DD on $NVDA: Strong earnings beat, 45% revenue growth YoY. P/E ratio still reasonable at 60. AI tailwinds should continue. My thesis is we see $600 by EOY.',
            'account_age_days': 500,
            'karma': 8000,
            'engagement': 120,
            'is_verified': False,
            'has_links': True,
            'subreddit': 'investing',
            'expected': True
        },
        {
            'name': 'IRRELEVANT: Product Review',
            'text': 'Just got my new iPhone from AAPL and it\'s amazing! The camera is so much better than my old phone. Highly recommend!',
            'account_age_days': 180,
            'karma': 1000,
            'engagement': 20,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'technology',
            'expected': False
        },
        {
            'name': 'IRRELEVANT: Job Posting',
            'text': 'GOOGL is hiring software engineers in Mountain View. Great opportunity to work on cutting-edge AI projects. Apply now!',
            'account_age_days': 90,
            'karma': 500,
            'engagement': 5,
            'is_verified': False,
            'has_links': True,
            'subreddit': 'jobs',
            'expected': False
        },
        {
            'name': 'SPAM: Pump Scheme',
            'text': '🚀🚀🚀 $AMC TO THE MOON!!! 100X GUARANTEED RETURNS!!! BUY NOW BEFORE ITS TOO LATE!!! CANT LOSE!!! 🚀🚀🚀',
            'account_age_days': 15,
            'karma': 50,
            'engagement': 200,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'wallstreetbets',
            'expected': False
        },
        {
            'name': 'LOW QUALITY: One Word',
            'text': 'nice',
            'account_age_days': 100,
            'karma': 200,
            'engagement': 2,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'stocks',
            'expected': False
        },
        {
            'name': 'RELEVANT: Fundamental News',
            'text': 'MSFT earnings call tomorrow. Expecting revenue beat based on strong Azure growth. If they guide up, stock could pop 5-10%. Holding my position.',
            'account_age_days': 1000,
            'karma': 20000,
            'engagement': 90,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'investing',
            'expected': True
        },
        {
            'name': 'BORDERLINE: Meme with context',
            'text': '$GME apes together strong 🦍 But seriously, the short interest is still high at 20%. Could be another squeeze if we get volume.',
            'account_age_days': 250,
            'karma': 3000,
            'engagement': 60,
            'is_verified': False,
            'has_links': False,
            'subreddit': 'wallstreetbets',
            'expected': True  # Has trading intent despite meme language
        },
        {
            'name': 'RELEVANT: High Quality from New Account',
            'text': 'Analysis of $META: Recent cost-cutting measures should improve margins by 15%. User growth rebounding in emerging markets. Fair value estimate $350-$375. Currently undervalued at $310.',
            'account_age_days': 45,  # New account
            'karma': 150,  # Low karma
            'engagement': 30,
            'is_verified': False,
            'has_links': True,
            'subreddit': 'securityanalysis',
            'expected': False  # Good content but account too new
        }
    ]
    
    # Run tests
    results = []
    for i, test in enumerate(test_posts, 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'-'*80}")
        print(f"Text: \"{test['text'][:100]}...\"" if len(test['text']) > 100 else f"Text: \"{test['text']}\"")
        
        # Apply filter
        result = filter_system.filter_post(
            text=test['text'],
            account_age_days=test['account_age_days'],
            karma_or_followers=test['karma'],
            engagement_score=min(100, (test['engagement'] / 100) * 100),
            is_verified=test['is_verified'],
            has_links=test['has_links'],
            subreddit=test['subreddit']
        )
        
        # Display results
        print(f"\nScores:")
        print(f"  Relevance Score: {result['relevance_score']}")
        print(f"  Quality Score: {result['quality_score']}/100")
        print(f"  Account Score: {result['account_score']}/100")
        print(f"  Confidence: {result['confidence_level']} ({result['confidence_percentage']}%)")
        
        print(f"\nRelevance Breakdown:")
        print(f"  Trading Intent: {result['relevance_breakdown']['trading_intent']}")
        print(f"  Price Analysis: {result['relevance_breakdown']['price_analysis']}")
        print(f"  Fundamental: {result['relevance_breakdown']['fundamental']}")
        print(f"  Due Diligence: {result['relevance_breakdown']['due_diligence']}")
        print(f"  Spam Penalty: {result['relevance_breakdown']['spam_penalty']}")
        print(f"  Off-Topic Penalty: {result['relevance_breakdown']['off_topic_penalty']}")
        
        # Check if passed
        passed = result['passes_filter']
        expected = test['expected']
        status = "✓ PASS" if passed == expected else "✗ FAIL"
        
        print(f"\nFilter Decision: {'ACCEPT' if passed else 'REJECT'}")
        print(f"Expected: {'ACCEPT' if expected else 'REJECT'}")
        print(f"Test Result: {status}")
        
        results.append({
            'test_name': test['name'],
            'passed_filter': passed,
            'expected': expected,
            'correct': passed == expected,
            'confidence': result['confidence_percentage'],
            'relevance': result['relevance_score'],
            'quality': result['quality_score'],
            'account': result['account_score']
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"Total Tests: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {correct_count/total_count*100:.1f}%")
    
    print(f"\n{'Test Name':<45} {'Result':<10} {'Confidence':<12}")
    print("-"*80)
    for r in results:
        result_str = "✓ PASS" if r['correct'] else "✗ FAIL"
        print(f"{r['test_name']:<45} {result_str:<10} {r['confidence']:>6}%")
    
    print(f"\n{'='*80}\n")
    
    # Save results
    with open('filter_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Test results saved to filter_test_results.json")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    
    print("\n\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80 + "\n")
    
    filter_system = RelevanceFilter()
    
    edge_cases = [
        {
            'name': 'Empty text',
            'text': '',
            'account_age_days': 365,
            'karma': 5000,
        },
        {
            'name': 'Very long text (>500 words)',
            'text': 'In-depth analysis of market conditions. ' * 100,
            'account_age_days': 365,
            'karma': 5000,
        },
        {
            'name': 'Brand new account (<1 day)',
            'text': 'Great DD on $AAPL. Strong buy signal here.',
            'account_age_days': 0,
            'karma': 0,
        },
        {
            'name': 'Multiple tickers in one post',
            'text': 'Portfolio update: Long $AAPL, $MSFT, $GOOGL. Short $TSLA. All positions looking good.',
            'account_age_days': 500,
            'karma': 10000,
        },
    ]
    
    for case in edge_cases:
        print(f"\n{'-'*80}")
        print(f"Edge Case: {case['name']}")
        print(f"{'-'*80}")
        
        result = filter_system.filter_post(
            text=case['text'],
            account_age_days=case['account_age_days'],
            karma_or_followers=case['karma'],
            engagement_score=50,
            is_verified=False,
            has_links=False,
            subreddit='stocks'
        )
        
        print(f"Filter Result: {'ACCEPT' if result['passes_filter'] else 'REJECT'}")
        print(f"Confidence: {result['confidence_percentage']}%")
        print(f"Quality Score: {result['quality_score']}/100")


if __name__ == "__main__":
    print("\nStock Sentiment Agent - Filter Testing\n")
    
    # Run main tests
    test_relevance_filter()
    
    # Run edge case tests
    test_edge_cases()
    
    print("\n✓ All tests complete!\n")
