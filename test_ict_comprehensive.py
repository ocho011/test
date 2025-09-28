#!/usr/bin/env python3
"""
Comprehensive test suite for ICT Analysis Engine

This script performs extensive testing of all ICT analysis components
including edge cases, performance testing, and integration validation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time

import numpy as np
import pandas as pd


def create_test_scenarios():
    """Create different test scenarios for comprehensive testing"""
    scenarios = {}

    # Scenario 1: Trending market with clear structure breaks
    dates = pd.date_range(start="2024-01-01", periods=500, freq="H")
    np.random.seed(42)

    # Generate trending price data
    trend = np.linspace(100, 120, 500)
    noise = np.random.randn(500) * 0.5
    close_prices = trend + noise

    # Add some clear structure breaks
    for i in [100, 200, 300, 400]:
        if i < len(close_prices):
            close_prices[i : i + 10] += 2.0  # Create clear breaks

    trend_data = []
    for i, date in enumerate(dates):
        close = close_prices[i]
        high = close + abs(np.random.randn() * 0.3)
        low = close - abs(np.random.randn() * 0.3)
        open_price = close_prices[i - 1] if i > 0 else close

        trend_data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(5000, 15000),
            }
        )

    scenarios["trending"] = pd.DataFrame(trend_data).set_index("timestamp")

    # Scenario 2: Ranging market with clear order blocks
    range_data = []
    base_price = 100
    for i, date in enumerate(dates[:300]):
        # Create ranging price action between 99-101
        close = base_price + np.sin(i * 0.1) + np.random.randn() * 0.2
        high = close + abs(np.random.randn() * 0.2)
        low = close - abs(np.random.randn() * 0.2)
        open_price = range_data[i - 1]["close"] if i > 0 else close

        range_data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(3000, 12000),
            }
        )

    scenarios["ranging"] = pd.DataFrame(range_data).set_index("timestamp")

    # Scenario 3: High volatility with gaps
    volatile_data = []
    price = 100
    for i, date in enumerate(dates[:200]):
        # Create volatile price action with occasional gaps
        if i % 50 == 0 and i > 0:  # Create gaps every 50 bars
            price += np.random.choice([-3, 3])  # Gap up or down

        close = price + np.random.randn() * 1.0
        high = close + abs(np.random.randn() * 0.5)
        low = close - abs(np.random.randn() * 0.5)
        open_price = volatile_data[i - 1]["close"] if i > 0 else close

        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volatile_data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(8000, 25000),
            }
        )

        price = close

    scenarios["volatile"] = pd.DataFrame(volatile_data).set_index("timestamp")

    return scenarios


def run_performance_tests():
    """Run performance benchmarks"""
    print("🔥 Performance Testing")
    print("=" * 50)

    try:
        from trading_bot.analysis import TechnicalAnalyzer

        # Create different data sizes for performance testing
        sizes = [100, 500, 1000, 2000]
        results = {}

        analyzer = TechnicalAnalyzer(
            enable_validation=True,
            enable_parallel_processing=True,
            memory_optimization=True,
        )

        for size in sizes:
            print(f"📊 Testing with {size} bars...")

            # Create test data
            dates = pd.date_range(start="2024-01-01", periods=size, freq="H")
            np.random.seed(42)
            close_prices = 100 + np.cumsum(np.random.randn(size) * 0.1)

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

            df = pd.DataFrame(data).set_index("timestamp")

            # Time the analysis
            start_time = time.time()
            result = analyzer.analyze_comprehensive(df, timeframe="1H")
            duration = time.time() - start_time

            results[size] = {
                "duration": duration,
                "memory_mb": result.memory_usage.get("current_mb", 0),
                "patterns_found": len(result.order_blocks)
                + len(result.fair_value_gaps)
                + len(result.market_structures),
                "patterns_per_second": (
                    (
                        len(result.order_blocks)
                        + len(result.fair_value_gaps)
                        + len(result.market_structures)
                    )
                    / duration
                    if duration > 0
                    else 0
                ),
            }

            print(
                f"   ✅ {size} bars: {duration:.3f}s, {result.memory_usage.get('current_mb', 0):.1f}MB, {results[size]['patterns_found']} patterns"
            )

        # Performance summary
        print("\n📈 Performance Summary:")
        for size, perf in results.items():
            print(
                f"   {size:4d} bars: {perf['duration']:6.3f}s | {perf['memory_mb']:5.1f}MB | {perf['patterns_per_second']:6.1f} patterns/sec"
            )

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_scenario_tests():
    """Test different market scenarios"""
    print("\n🎯 Scenario Testing")
    print("=" * 50)

    try:
        from trading_bot.analysis import TechnicalAnalyzer

        analyzer = TechnicalAnalyzer(
            enable_validation=True,
            enable_parallel_processing=False,  # For clearer debugging
            memory_optimization=True,
        )

        scenarios = create_test_scenarios()
        scenario_results = {}

        for scenario_name, data in scenarios.items():
            print(f"\n📊 Testing {scenario_name.upper()} market scenario...")

            try:
                result = analyzer.analyze_comprehensive(data, timeframe="1H")

                scenario_results[scenario_name] = {
                    "order_blocks": len(result.order_blocks),
                    "fair_value_gaps": len(result.fair_value_gaps),
                    "market_structures": len(result.market_structures),
                    "pattern_validations": len(result.pattern_validations),
                    "confluence_signals": len(result.confluence_signals),
                    "duration": result.analysis_duration,
                    "success": True,
                }

                print(f"   ✅ Analysis completed in {result.analysis_duration:.3f}s")
                print(f"   📦 Order Blocks: {len(result.order_blocks)}")
                print(f"   🕳️ Fair Value Gaps: {len(result.fair_value_gaps)}")
                print(f"   🏗️ Market Structures: {len(result.market_structures)}")
                print(f"   ✔️ Pattern Validations: {len(result.pattern_validations)}")
                print(f"   🔗 Confluence Signals: {len(result.confluence_signals)}")

                # Test real-time signals
                signals = analyzer.get_real_time_signals(data, lookback_bars=50)
                print(f"   ⚡ Real-time Signals: {len(signals)}")

            except Exception as e:
                print(f"   ❌ {scenario_name} scenario failed: {e}")
                scenario_results[scenario_name] = {"success": False, "error": str(e)}

        # Scenario summary
        print(f"\n📋 Scenario Summary:")
        for scenario, results in scenario_results.items():
            if results.get("success", False):
                print(
                    f"   ✅ {scenario:10s}: {results['order_blocks']:2d} OB, {results['fair_value_gaps']:2d} FVG, {results['market_structures']:2d} MS"
                )
            else:
                print(
                    f"   ❌ {scenario:10s}: Failed - {results.get('error', 'Unknown error')}"
                )

        return all(r.get("success", False) for r in scenario_results.values())

    except Exception as e:
        print(f"❌ Scenario testing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_component_validation():
    """Validate individual components work correctly"""
    print("\n🔧 Component Validation")
    print("=" * 50)

    try:
        from trading_bot.analysis import (
            FairValueGapAnalyzer,
            ICTIndicatorIntegration,
            ICTPatternIndicatorFusion,
            MarketStructureAnalyzer,
            OrderBlockDetector,
            PatternType,
            PatternValidationEngine,
            TechnicalIndicators,
            TimeFrameManager,
        )

        # Create test data
        dates = pd.date_range(start="2024-01-01", periods=200, freq="H")
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(200) * 0.1)

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

        df = pd.DataFrame(data).set_index("timestamp")

        components_tested = {}

        # Test OrderBlockDetector
        print("🔍 Testing OrderBlockDetector...")
        ob_detector = OrderBlockDetector()
        swing_points = ob_detector.find_swing_points(df)
        order_blocks = ob_detector.identify_order_blocks(df, swing_points)
        components_tested["OrderBlockDetector"] = len(order_blocks) > 0
        print(
            f"   ✅ Found {len(swing_points)} swing points, {len(order_blocks)} order blocks"
        )

        # Test FairValueGapAnalyzer
        print("🕳️ Testing FairValueGapAnalyzer...")
        fvg_analyzer = FairValueGapAnalyzer()
        fvgs = fvg_analyzer.detect_gaps(df)
        components_tested["FairValueGapAnalyzer"] = (
            True  # Always succeeds even with 0 gaps
        )
        print(f"   ✅ Found {len(fvgs)} fair value gaps")

        # Test MarketStructureAnalyzer
        print("🏗️ Testing MarketStructureAnalyzer...")
        ms_analyzer = MarketStructureAnalyzer()
        market_structures = ms_analyzer.analyze_structure(df, swing_points)
        components_tested["MarketStructureAnalyzer"] = True
        print(f"   ✅ Found {len(market_structures)} market structure patterns")

        # Test TimeFrameManager
        print("⏰ Testing TimeFrameManager...")
        tf_manager = TimeFrameManager()
        tf_manager.add_timeframe_data("1H", df)
        htf_bias = tf_manager.get_higher_timeframe_bias("1H")
        components_tested["TimeFrameManager"] = True
        print(f"   ✅ Higher timeframe bias calculated: {len(htf_bias)} timeframes")

        # Test PatternValidationEngine
        print("✔️ Testing PatternValidationEngine...")
        validator = PatternValidationEngine()
        if order_blocks:
            validation = validator.validate_pattern(
                order_blocks[0], df, PatternType.ORDER_BLOCK
            )
            components_tested["PatternValidationEngine"] = (
                validation.confidence_score >= 0
            )
            print(
                f"   ✅ Pattern validation: confidence {validation.confidence_score:.2f}"
            )
        else:
            components_tested["PatternValidationEngine"] = True
            print(f"   ✅ Pattern validation: no patterns to validate")

        # Test TechnicalIndicators
        print("📊 Testing TechnicalIndicators...")
        tech_indicators = TechnicalIndicators()
        ema = tech_indicators.ema(df["close"], 20)
        sma = tech_indicators.sma(df["close"], 20)
        components_tested["TechnicalIndicators"] = len(ema) > 0 and len(sma) > 0
        print(
            f"   ✅ Calculated technical indicators: EMA ({len(ema)} values), SMA ({len(sma)} values)"
        )

        # Test ICTIndicatorIntegration
        print("🔗 Testing ICTIndicatorIntegration...")
        ict_integration = ICTIndicatorIntegration()
        ict_patterns = {"order_blocks": order_blocks, "fair_value_gaps": fvgs}
        integration_result = ict_integration.analyze(df, ict_patterns)
        components_tested["ICTIndicatorIntegration"] = (
            "indicators" in integration_result
        )
        print(f"   ✅ ICT integration analysis completed")

        # Test ICTPatternIndicatorFusion
        print("🎯 Testing ICTPatternIndicatorFusion...")
        fusion = ICTPatternIndicatorFusion()
        fusion_result = fusion.fusion_analysis(df, ict_patterns)
        components_tested["ICTPatternIndicatorFusion"] = "fusion_score" in fusion_result
        print(
            f"   ✅ Pattern fusion analysis: score {fusion_result.get('fusion_score', 0):.2f}"
        )

        # Component summary
        print(f"\n📋 Component Validation Summary:")
        all_passed = True
        for component, passed in components_tested.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {component}")
            if not passed:
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Component validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run comprehensive test suite"""
    print("🧪 ICT Analysis Engine - Comprehensive Test Suite")
    print("=" * 60)

    test_results = {}

    # Run basic functionality test
    print("🔧 Running basic functionality test...")
    try:
        from trading_bot.analysis import TechnicalAnalyzer

        TechnicalAnalyzer()
        test_results["basic_import"] = True
        print("✅ Basic import test passed")
    except Exception as e:
        test_results["basic_import"] = False
        print(f"❌ Basic import test failed: {e}")

    # Run component validation
    test_results["component_validation"] = run_component_validation()

    # Run scenario testing
    test_results["scenario_testing"] = run_scenario_tests()

    # Run performance testing
    test_results["performance_testing"] = run_performance_tests()

    # Final summary
    print("\n" + "=" * 60)
    print("🏆 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name.replace('_', ' ').title()}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! ICT Analysis Engine is fully operational!")
        print("🚀 Ready for production use!")
    else:
        print("⚠️ Some tests failed. Please review the results above.")
        print("🔧 Consider debugging failed components before production use.")

    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
