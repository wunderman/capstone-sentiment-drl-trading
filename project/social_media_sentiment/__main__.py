"""
Entry point for running the social media sentiment pipeline.
Usage: python -m social_media_sentiment [command]

Commands:
  dataset   - Run the dataset-based pipeline (no APIs needed)
  analyze   - Run live analysis for a ticker (Telegram scraping)
  live      - Run full agentic analysis via Apify (Twitter + Reddit + Telegram)
  test      - Run offline tests
"""

import sys


def main():
    args = sys.argv[1:]
    command = args[0] if args else "dataset"

    if command == "dataset":
        from .dataset_collector import main as dataset_main
        dataset_main()

    elif command == "analyze":
        ticker = args[1] if len(args) > 1 else "TSLA"
        from .stock_sentiment_agent import StockSentimentAgent
        agent = StockSentimentAgent(
            use_reddit=False, use_twitter=False, use_stocktwits=False,
            use_telegram=True, use_youtube=False, use_bluesky=False
        )
        result = agent.analyze_ticker(ticker)
        if result:
            print(f"\nSignal: {result['signal']} | Score: {result['sentiment_score']:.4f}")
            print(f"Posts analyzed: {result['total_posts']}")

    elif command == "live":
        ticker = args[1] if len(args) > 1 else "TSLA"
        limit = int(args[2]) if len(args) > 2 else 50
        from .stock_sentiment_agent import StockSentimentAgent
        agent = StockSentimentAgent(
            use_reddit=False,
            use_twitter=False,
            use_stocktwits=False,
            use_telegram=False,
            use_youtube=False,
            use_bluesky=False,
            use_apify=True,
            use_llm_filter=True,
        )
        result = agent.analyze_ticker(ticker, apify_limit=limit, hours_back=48)
        if result and result.get('trade_signal'):
            signal = result['trade_signal']
            print(f"\n{'='*50}")
            print(f"  {ticker}  \u2192  {signal['action']}")
            print(f"  Sentiment: {signal['sentiment_score']:+.4f}")
            print(f"  Strength:  {signal['signal_strength']:.0f}/100")
            print(f"  Reliability: {signal['reliability']:.0f}/100")
            print(f"  Posts: {result['posts_passed_filter']} filtered / {result['total_posts_collected']} collected")
            print(f"{'='*50}")

    elif command == "test":
        from .tests.test_agent_offline import test_without_apis
        test_without_apis()

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
