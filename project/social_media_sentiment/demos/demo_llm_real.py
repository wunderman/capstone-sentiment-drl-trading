"""
Real-world LLM vs Regex Ticker Extraction Demo
Shows actual Telegram messages and how LLM processes them vs regex
"""

from ..collectors.telegram_collector import TelegramCollector
from ..llm_ticker_extractor import OllamaTickerExtractor
import re


print("\n" + "="*80)
print("  REAL-WORLD: LLM vs REGEX ON ACTUAL TELEGRAM MESSAGES")
print("="*80)

# Initialize
print("\n[Step 1] Initializing Ollama LLM...")
llm = OllamaTickerExtractor()

print("\n[Step 2] Scraping 50 recent Telegram messages...")
collector = TelegramCollector(use_llm=False)
messages = collector.scrape_all_messages(limit=50)
print(f"✓ Got {len(messages)} messages\n")

# Process first 5 messages with substantial content
count = 0
for msg in messages:
    if len(msg['text']) > 50:  # Skip very short messages
        count += 1
        text = msg['text']
        
        print("\n" + "="*80)
        print(f"MESSAGE #{count} from @{msg['channel']}")
        print("="*80)
        print(f"{text[:500]}..." if len(text) > 500 else text)
        print("\n" + "-"*80)
        
        # REGEX EXTRACTION
        regex_tickers = set()
        # Pattern 1: $SYMBOL
        dollar = re.findall(r'\$([A-Z]{2,5})\b', text)
        regex_tickers.update(dollar)
        # Pattern 2: SYMBOL.NS/BO
        indian = re.findall(r'\b([A-Z]+)\.(NS|BO)\b', text)
        regex_tickers.update([t[0] for t in indian])
        # Pattern 3: Known tickers
        known = {'HDFC', 'ICICI', 'SBI', 'BAJAJ', 'NTPC', 'ADANI', 'RELIANCE', 'TCS', 'INFY', 'REC', 'PFC', 'BPCL', 'HINDALCO', 'ITC', 'BHARTI'}
        for ticker in known:
            if re.search(rf'\b{ticker}\b', text, re.IGNORECASE):
                regex_tickers.add(ticker)
        
        print(f"\n📊 REGEX EXTRACTION:")
        print(f"   Patterns: $SYMBOL, SYMBOL.NS/BO, Known tickers")
        print(f"   Found: {sorted(list(regex_tickers)) if regex_tickers else 'None'}")
        
        # LLM EXTRACTION
        print(f"\n🤖 LLM EXTRACTION (Ollama llama3.2):")
        print(f"   Analyzing context... (takes ~5-10 seconds)")
        llm_tickers = llm.extract_tickers(text)
        print(f"   Found: {llm_tickers if llm_tickers else 'None'}")
        
        # COMPARISON
        print(f"\n🔍 COMPARISON:")
        if set(regex_tickers) == set(llm_tickers):
            print(f"   ✓ Both methods agree: {sorted(list(regex_tickers))}")
        else:
            only_regex = set(regex_tickers) - set(llm_tickers)
            only_llm = set(llm_tickers) - set(regex_tickers)
            both = set(regex_tickers) & set(llm_tickers)
            
            if both:
                print(f"   ✓ Both found: {sorted(list(both))}")
            if only_regex:
                print(f"   ⚠️  Regex found but LLM filtered: {sorted(list(only_regex))}")
                print(f"       → LLM likely detected these are not being actively discussed")
            if only_llm:
                print(f"   💡 LLM found but Regex missed: {sorted(list(only_llm))}")
                print(f"       → LLM inferred these from context")
        
        if count >= 5:
            break
        
        input("\n👉 Press Enter for next message...")

print("\n" + "="*80)
print("  ANALYSIS COMPLETE")
print("="*80)
print("\n💡 Key Takeaways:")
print("   • LLM understands context and filters negative mentions")
print("   • LLM can infer tickers not explicitly marked with $ or .NS")
print("   • Regex is faster and more consistent with explicit patterns")
print("   • Best approach: Hybrid - use both and validate against each other")
print()
