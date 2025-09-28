#!/usr/bin/env python3
"""
Quick test script for ICT Analysis Engine

This script tests the basic functionality of the ICT analysis engine
to ensure all components are working correctly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd


def create_sample_data(bars=200):
    """Create sample OHLC data for testing"""
    dates = pd.date_range(start="2024-01-01", periods=bars, freq="H")

    # Generate realistic OHLC data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(bars) * 0.1)

    data = []
    for i, date in enumerate(dates):
        close = close_prices[i]
        high = close + abs(np.random.randn() * 0.2)
        low = close - abs(np.random.randn() * 0.2)
        open_price = close_prices[i - 1] if i > 0 else close

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000),
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def test_ict_engine():
    """Test the ICT Analysis Engine"""
    print("🧪 Testing ICT Analysis Engine")
    print("=" * 50)

    try:
        # Import the main analyzer
        from trading_bot.analysis import TechnicalAnalyzer

        print("✅ TechnicalAnalyzer imported successfully")

        # Create sample data
        print("\n📊 Creating sample OHLC data...")
        data = create_sample_data(200)
        print(f"✅ Created {len(data)} bars of sample data")

        # Initialize analyzer
        print("\n🔧 Initializing TechnicalAnalyzer...")
        analyzer = TechnicalAnalyzer(
            enable_validation=True,
            enable_parallel_processing=False,  # Disable for testing
            memory_optimization=True,
            performance_monitoring=True,
        )
        print("✅ TechnicalAnalyzer initialized successfully")

        # Test individual components
        print("\n🔍 Testing individual components...")

        # Test Order Block detection
        swing_points = analyzer.order_block_detector.find_swing_points(data)
        order_blocks = analyzer.order_block_detector.identify_order_blocks(
            data, swing_points
        )
        print(f"✅ Order Blocks: Found {len(order_blocks)} patterns")

        # Test Fair Value Gap analysis
        fvgs = analyzer.fvg_analyzer.detect_gaps(data)
        print(f"✅ Fair Value Gaps: Found {len(fvgs)} patterns")

        # Test Market Structure analysis
        market_structures = analyzer.market_structure_analyzer.analyze_structure(
            data, swing_points
        )
        print(f"✅ Market Structures: Found {len(market_structures)} patterns")

        # Test Technical Indicators
        indicators = analyzer.ict_indicator_integration.calculate_all_indicators(data)
        print(f"✅ Technical Indicators: Calculated {len(indicators)} indicators")

        # Test comprehensive analysis
        print("\n🚀 Testing comprehensive analysis...")
        result = analyzer.analyze_comprehensive(data, timeframe="1H")

        print(f"✅ Comprehensive Analysis completed in {result.analysis_duration:.3f}s")
        print(f"   - Order Blocks: {len(result.order_blocks)}")
        print(f"   - Fair Value Gaps: {len(result.fair_value_gaps)}")
        print(f"   - Market Structures: {len(result.market_structures)}")
        print(f"   - Pattern Validations: {len(result.pattern_validations)}")
        print(f"   - Confluence Signals: {len(result.confluence_signals)}")
        print(f"   - Memory Usage: {result.memory_usage.get('current_mb', 0):.1f}MB")

        # Test real-time signals
        print("\n⚡ Testing real-time signals...")
        signals = analyzer.get_real_time_signals(data, lookback_bars=50)
        print(f"✅ Real-time Signals: Generated {len(signals)} signals")

        # Test performance report
        print("\n📈 Testing performance report...")
        perf_report = analyzer.get_performance_report()
        print(
            f"✅ Performance Report: {perf_report.get('total_analyses', 0)} analyses recorded"
        )

        print("\n🎉 All tests passed successfully!")
        print("=" * 50)
        print("🏆 ICT Analysis Engine is working correctly!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ict_engine()
    sys.exit(0 if success else 1)
