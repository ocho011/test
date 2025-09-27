"""
Main Technical Analyzer for ICT Analysis Engine

This module provides the TechnicalAnalyzer class that integrates all ICT analysis components:
- OrderBlockDetector, FairValueGapAnalyzer, MarketStructureAnalyzer, TimeFrameManager
- Technical indicators integration
- Pattern validation system
- Performance optimization with numpy vectorization
- Memory management and garbage collection optimization
- Real-time analysis performance benchmarking
"""

import gc
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from .ict_analyzer import (
    FairValueGap,
    FairValueGapAnalyzer,
    MarketStructure,
    MarketStructureAnalyzer,
    OrderBlock,
    OrderBlockDetector,
    PatternType,
    PatternValidationEngine,
    PatternValidationResult,
    TimeFrameManager,
)
from .technical_indicators import (
    ICTIndicatorIntegration,
    ICTPatternIndicatorFusion,
    TechnicalIndicators,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result with all ICT patterns and indicators"""

    timestamp: pd.Timestamp
    timeframe: str

    # ICT Patterns
    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    market_structures: List[MarketStructure]

    # Technical Indicators
    indicators: Dict[str, Any]

    # Pattern Validation
    pattern_validations: List[PatternValidationResult]

    # Confluence Analysis
    confluence_signals: List[Dict[str, Any]]

    # Performance Metrics
    analysis_duration: float
    memory_usage: Dict[str, float]

    # Summary Statistics
    summary: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance benchmarking metrics"""

    total_analysis_time: float
    component_times: Dict[str, float]
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    patterns_per_second: float
    data_size: int


class TechnicalAnalyzer:
    """
    Main Technical Analyzer for ICT Analysis Engine

    Integrates all ICT analysis components with performance optimization:
    - Order Block detection and analysis
    - Fair Value Gap identification and tracking
    - Market Structure analysis (BOS/CHoCH)
    - Multi-timeframe analysis coordination
    - Technical indicators integration
    - Pattern validation and reliability assessment
    - Performance optimization and memory management
    """

    def __init__(
        self,
        enable_validation: bool = True,
        enable_parallel_processing: bool = True,
        memory_optimization: bool = True,
        performance_monitoring: bool = True,
    ):
        """
        Initialize Technical Analyzer

        Args:
            enable_validation: Enable pattern validation system
            enable_parallel_processing: Enable parallel processing for performance
            memory_optimization: Enable memory optimization features
            performance_monitoring: Enable performance monitoring and benchmarking
        """
        self.enable_validation = enable_validation
        self.enable_parallel_processing = enable_parallel_processing
        self.memory_optimization = memory_optimization
        self.performance_monitoring = performance_monitoring

        # Initialize ICT components
        self.order_block_detector = OrderBlockDetector()
        self.fvg_analyzer = FairValueGapAnalyzer()
        self.market_structure_analyzer = MarketStructureAnalyzer()
        self.timeframe_manager = TimeFrameManager()

        # Initialize technical indicators
        self.technical_indicators = TechnicalIndicators()
        self.ict_indicator_integration = ICTIndicatorIntegration()
        self.pattern_indicator_fusion = ICTPatternIndicatorFusion()

        # Initialize validation engine
        if self.enable_validation:
            self.validation_engine = PatternValidationEngine()

        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.analysis_count = 0

        # Memory management
        self._memory_threshold_mb = 500  # Trigger cleanup at 500MB

        logger.info("TechnicalAnalyzer initialized with all ICT components")

    def analyze_comprehensive(
        self,
        data: pd.DataFrame,
        timeframe: str = "1H",
        enable_validation: Optional[bool] = None,
        max_workers: int = 4,
    ) -> AnalysisResult:
        """
        Perform comprehensive ICT analysis on the provided data

        Args:
            data: OHLC DataFrame with required columns
            timeframe: Analysis timeframe
            enable_validation: Override validation setting
            max_workers: Maximum worker threads for parallel processing

        Returns:
            Complete analysis result with all patterns and indicators
        """
        start_time = time.time()
        analysis_start_memory = self._get_memory_usage()

        try:
            logger.info(
                f"Starting comprehensive ICT analysis for {len(data)} bars on {timeframe}"
            )

            # Validate input data
            self._validate_input_data(data)

            # Memory optimization: Clean up before large analysis
            if self.memory_optimization:
                self._optimize_memory()

            # Initialize result containers
            results = {
                "order_blocks": [],
                "fair_value_gaps": [],
                "market_structures": [],
                "indicators": {},
                "pattern_validations": [],
                "confluence_signals": [],
                "component_times": {},
            }

            # Run analysis components
            if self.enable_parallel_processing and len(data) > 100:
                results = self._run_parallel_analysis(data, results, max_workers)
            else:
                results = self._run_sequential_analysis(data, results)

            # Perform confluence analysis
            confluence_start = time.time()
            confluence_signals = self._analyze_confluence(results, data)
            results["confluence_signals"] = confluence_signals
            results["component_times"]["confluence"] = time.time() - confluence_start

            # Calculate performance metrics
            analysis_duration = time.time() - start_time
            memory_usage = self._calculate_memory_metrics(analysis_start_memory)

            # Create comprehensive result
            analysis_result = AnalysisResult(
                timestamp=pd.Timestamp.now(),
                timeframe=timeframe,
                order_blocks=results["order_blocks"],
                fair_value_gaps=results["fair_value_gaps"],
                market_structures=results["market_structures"],
                indicators=results["indicators"],
                pattern_validations=results["pattern_validations"],
                confluence_signals=confluence_signals,
                analysis_duration=analysis_duration,
                memory_usage=memory_usage,
                summary=self._generate_analysis_summary(results, analysis_duration),
            )

            # Performance monitoring
            if self.performance_monitoring:
                self._record_performance_metrics(
                    analysis_result, results["component_times"], len(data)
                )

            # Memory cleanup
            if self.memory_optimization:
                self._cleanup_after_analysis()

            self.analysis_count += 1
            logger.info(f"Comprehensive analysis complete in {analysis_duration:.3f}s")

            return analysis_result

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise

    def analyze_single_timeframe(
        self, data: pd.DataFrame, timeframe: str, components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze single timeframe with specified components

        Args:
            data: OHLC DataFrame
            timeframe: Analysis timeframe
            components: List of components to run ['order_blocks', 'fvg', 'structure', 'indicators']

        Returns:
            Analysis results for specified components
        """
        if components is None:
            components = ["order_blocks", "fvg", "structure", "indicators"]

        start_time = time.time()
        results = {}

        try:
            logger.info(
                f"Single timeframe analysis: {timeframe} with components: {components}"
            )

            # Order Block Detection
            if "order_blocks" in components:
                swing_points = self.order_block_detector.find_swing_points(data)
                order_blocks = self.order_block_detector.identify_order_blocks(
                    data, swing_points
                )
                results["order_blocks"] = order_blocks
                logger.debug(f"Found {len(order_blocks)} order blocks")

            # Fair Value Gap Analysis
            if "fvg" in components:
                fair_value_gaps = self.fvg_analyzer.detect_gaps(data)
                results["fair_value_gaps"] = fair_value_gaps
                logger.debug(f"Found {len(fair_value_gaps)} fair value gaps")

            # Market Structure Analysis
            if "structure" in components:
                swing_points = results.get(
                    "swing_points", self.order_block_detector.find_swing_points(data)
                )
                market_structures = self.market_structure_analyzer.analyze_structure(
                    data, swing_points
                )
                results["market_structures"] = market_structures
                logger.debug(
                    f"Found {len(market_structures)} market structure patterns"
                )

            # Technical Indicators
            if "indicators" in components:
                indicators = self.ict_indicator_integration.calculate_all_indicators(
                    data
                )
                results["indicators"] = indicators
                logger.debug(f"Calculated {len(indicators)} technical indicators")

            duration = time.time() - start_time
            results["analysis_duration"] = duration
            results["timeframe"] = timeframe

            return results

        except Exception as e:
            logger.error(f"Error in single timeframe analysis: {e}")
            raise

    def get_real_time_signals(
        self, data: pd.DataFrame, lookback_bars: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get real-time trading signals based on current market conditions

        Args:
            data: Latest OHLC data
            lookback_bars: Number of bars to analyze for signals

        Returns:
            List of real-time trading signals
        """
        try:
            # Use latest data for real-time analysis
            recent_data = data.tail(lookback_bars).copy()

            # Fast analysis for real-time signals
            signals = []

            # Quick Order Block signals
            swing_points = self.order_block_detector.find_swing_points(recent_data)
            if swing_points:
                latest_ob = self.order_block_detector.identify_order_blocks(
                    recent_data, swing_points
                )
                if latest_ob:
                    ob_signal = self._generate_order_block_signal(
                        latest_ob[-1], recent_data
                    )
                    if ob_signal:
                        signals.append(ob_signal)

            # Quick FVG signals
            fvgs = self.fvg_analyzer.detect_gaps(recent_data)
            if fvgs:
                for fvg in fvgs[-3:]:  # Check last 3 FVGs
                    if not fvg.is_filled:
                        fvg_signal = self._generate_fvg_signal(fvg, recent_data)
                        if fvg_signal:
                            signals.append(fvg_signal)

            # Quick structure break signals
            structures = self.market_structure_analyzer.detect_structural_breaks(
                recent_data
            )
            for structure_type, structure_list in structures.items():
                if structure_list:
                    structure_signal = self._generate_structure_signal(
                        structure_list[-1], recent_data
                    )
                    if structure_signal:
                        signals.append(structure_signal)

            # Filter and rank signals
            signals = self._filter_and_rank_signals(signals, recent_data)

            logger.info(f"Generated {len(signals)} real-time signals")
            return signals

        except Exception as e:
            logger.error(f"Error generating real-time signals: {e}")
            return []

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis report"""
        if not self.performance_history:
            return {"error": "No performance data available"}

        try:
            # Calculate performance statistics
            avg_analysis_time = np.mean(
                [p.total_analysis_time for p in self.performance_history]
            )
            avg_memory_usage = np.mean(
                [p.memory_usage_mb for p in self.performance_history]
            )
            avg_patterns_per_second = np.mean(
                [p.patterns_per_second for p in self.performance_history]
            )

            # Component performance breakdown
            component_stats = {}
            for metric in self.performance_history:
                for component, time_taken in metric.component_times.items():
                    if component not in component_stats:
                        component_stats[component] = []
                    component_stats[component].append(time_taken)

            component_averages = {
                comp: np.mean(times) for comp, times in component_stats.items()
            }

            return {
                "total_analyses": len(self.performance_history),
                "average_analysis_time": avg_analysis_time,
                "average_memory_usage_mb": avg_memory_usage,
                "average_patterns_per_second": avg_patterns_per_second,
                "component_performance": component_averages,
                "memory_optimization_enabled": self.memory_optimization,
                "parallel_processing_enabled": self.enable_parallel_processing,
                "validation_enabled": self.enable_validation,
            }

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

    # Private helper methods

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input OHLC data"""
        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < 50:
            logger.warning(f"Limited data: only {len(data)} bars available")

        # Check for data quality issues
        if data[required_columns].isnull().any().any():
            logger.warning("Data contains null values, analysis may be affected")

    def _run_parallel_analysis(
        self, data: pd.DataFrame, results: Dict, max_workers: int
    ) -> Dict:
        """Run analysis components in parallel for better performance"""
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit parallel tasks
                futures = {}

                # Order Block detection
                futures["order_blocks"] = executor.submit(
                    self._analyze_order_blocks, data
                )

                # Fair Value Gap analysis
                futures["fvg"] = executor.submit(self._analyze_fair_value_gaps, data)

                # Market Structure analysis (needs swing points, so run after OB)
                futures["structure"] = executor.submit(
                    self._analyze_market_structure, data
                )

                # Technical Indicators
                futures["indicators"] = executor.submit(self._analyze_indicators, data)

                # Collect results
                for component, future in futures.items():
                    try:
                        component_result = future.result(
                            timeout=30
                        )  # 30 second timeout
                        results.update(component_result)
                    except Exception as e:
                        logger.error(f"Error in parallel {component} analysis: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            # Fallback to sequential analysis
            return self._run_sequential_analysis(data, results)

    def _run_sequential_analysis(self, data: pd.DataFrame, results: Dict) -> Dict:
        """Run analysis components sequentially"""
        try:
            # Order Block Analysis
            ob_result = self._analyze_order_blocks(data)
            results.update(ob_result)

            # Fair Value Gap Analysis
            fvg_result = self._analyze_fair_value_gaps(data)
            results.update(fvg_result)

            # Market Structure Analysis
            structure_result = self._analyze_market_structure(data)
            results.update(structure_result)

            # Technical Indicators
            indicators_result = self._analyze_indicators(data)
            results.update(indicators_result)

            return results

        except Exception as e:
            logger.error(f"Error in sequential analysis: {e}")
            raise

    def _analyze_order_blocks(self, data: pd.DataFrame) -> Dict:
        """Analyze Order Blocks with timing"""
        start_time = time.time()
        try:
            swing_points = self.order_block_detector.find_swing_points(data)
            order_blocks = self.order_block_detector.identify_order_blocks(
                data, swing_points
            )

            # Pattern validation if enabled
            validations = []
            if self.enable_validation and order_blocks:
                for ob in order_blocks[-5:]:  # Validate last 5 patterns
                    validation = self.validation_engine.validate_pattern(
                        ob, data, PatternType.ORDER_BLOCK
                    )
                    validations.append(validation)

            duration = time.time() - start_time
            return {
                "order_blocks": order_blocks,
                "swing_points": swing_points,
                "pattern_validations": validations,
                "component_times": {"order_blocks": duration},
            }

        except Exception as e:
            logger.error(f"Error analyzing order blocks: {e}")
            return {"order_blocks": [], "swing_points": [], "pattern_validations": []}

    def _analyze_fair_value_gaps(self, data: pd.DataFrame) -> Dict:
        """Analyze Fair Value Gaps with timing"""
        start_time = time.time()
        try:
            fair_value_gaps = self.fvg_analyzer.detect_gaps(data)

            # Update gap status
            fair_value_gaps = self.fvg_analyzer.update_gap_status(fair_value_gaps, data)

            # Pattern validation if enabled
            validations = []
            if self.enable_validation and fair_value_gaps:
                for fvg in fair_value_gaps[-3:]:  # Validate last 3 patterns
                    validation = self.validation_engine.validate_pattern(
                        fvg, data, PatternType.FAIR_VALUE_GAP
                    )
                    validations.append(validation)

            duration = time.time() - start_time
            return {
                "fair_value_gaps": fair_value_gaps,
                "fvg_validations": validations,
                "component_times": {"fair_value_gaps": duration},
            }

        except Exception as e:
            logger.error(f"Error analyzing fair value gaps: {e}")
            return {"fair_value_gaps": [], "fvg_validations": []}

    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze Market Structure with timing"""
        start_time = time.time()
        try:
            # Get swing points (reuse if available)
            swing_points = self.order_block_detector.find_swing_points(data)
            market_structures = self.market_structure_analyzer.analyze_structure(
                data, swing_points
            )

            # Detect structural breaks
            structural_breaks = self.market_structure_analyzer.detect_structural_breaks(
                data
            )

            # Pattern validation if enabled
            validations = []
            if self.enable_validation and market_structures:
                for ms in market_structures[-3:]:  # Validate last 3 patterns
                    pattern_type = (
                        PatternType.BREAK_OF_STRUCTURE
                        if ms.pattern_type == "BOS"
                        else PatternType.CHANGE_OF_CHARACTER
                    )
                    validation = self.validation_engine.validate_pattern(
                        ms, data, pattern_type
                    )
                    validations.append(validation)

            duration = time.time() - start_time
            return {
                "market_structures": market_structures,
                "structural_breaks": structural_breaks,
                "structure_validations": validations,
                "component_times": {"market_structure": duration},
            }

        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {
                "market_structures": [],
                "structural_breaks": {},
                "structure_validations": [],
            }

    def _analyze_indicators(self, data: pd.DataFrame) -> Dict:
        """Analyze Technical Indicators with timing"""
        start_time = time.time()
        try:
            # Calculate all indicators
            indicators = self.ict_indicator_integration.calculate_all_indicators(data)

            duration = time.time() - start_time
            return {
                "indicators": indicators,
                "component_times": {"indicators": duration},
            }

        except Exception as e:
            logger.error(f"Error analyzing indicators: {e}")
            return {"indicators": {}}

    def _analyze_confluence(
        self, results: Dict, data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Analyze confluence between different patterns and indicators"""
        try:
            confluence_signals = []

            # ICT pattern confluence
            ict_patterns = {
                "order_blocks": results.get("order_blocks", []),
                "fair_value_gaps": results.get("fair_value_gaps", []),
                "market_structures": results.get("market_structures", []),
            }

            indicators = results.get("indicators", {})

            # Use pattern indicator fusion for confluence analysis
            if indicators and any(ict_patterns.values()):
                fusion_analysis = self.pattern_indicator_fusion.fusion_analysis(
                    data, ict_patterns
                )

                # Extract fusion signals from the analysis
                fusion_signals = fusion_analysis.get("fusion_signals", [])

                # Convert fusion signals to confluence signals
                for signal in fusion_signals:
                    if signal and signal.get("strength", 0) > 0.6:
                        confluence_signals.append(
                            {
                                "type": signal.get("signal_type", "fusion_signal"),
                                "strength": signal.get("strength", 0),
                                "components": signal.get("factors", []),
                                "timestamp": signal.get(
                                    "timestamp", pd.Timestamp.now()
                                ),
                                "confidence": fusion_analysis.get("fusion_score", 0),
                            }
                        )

            return confluence_signals

        except Exception as e:
            logger.error(f"Error in confluence analysis: {e}")
            return []

    def _generate_analysis_summary(
        self, results: Dict, duration: float
    ) -> Dict[str, Any]:
        """Generate summary statistics for the analysis"""
        try:
            summary = {
                "analysis_duration": duration,
                "patterns_found": {
                    "order_blocks": len(results.get("order_blocks", [])),
                    "fair_value_gaps": len(results.get("fair_value_gaps", [])),
                    "market_structures": len(results.get("market_structures", [])),
                },
                "confluence_signals": len(results.get("confluence_signals", [])),
                "indicators_calculated": len(results.get("indicators", {})),
                "patterns_validated": len(results.get("pattern_validations", [])),
                "component_times": results.get("component_times", {}),
                "memory_optimized": self.memory_optimization,
                "parallel_processed": self.enable_parallel_processing,
            }

            # Calculate performance score
            patterns_found = sum(summary["patterns_found"].values())
            if duration > 0:
                summary["patterns_per_second"] = patterns_found / duration
            else:
                summary["patterns_per_second"] = 0

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

    def _record_performance_metrics(
        self, result: AnalysisResult, component_times: Dict, data_size: int
    ) -> None:
        """Record performance metrics for monitoring"""
        try:
            patterns_found = (
                len(result.order_blocks)
                + len(result.fair_value_gaps)
                + len(result.market_structures)
            )

            patterns_per_second = (
                patterns_found / result.analysis_duration
                if result.analysis_duration > 0
                else 0
            )

            metrics = PerformanceMetrics(
                total_analysis_time=result.analysis_duration,
                component_times=component_times,
                memory_usage_mb=result.memory_usage.get("current_mb", 0),
                memory_peak_mb=result.memory_usage.get("peak_mb", 0),
                cpu_usage_percent=0.0,  # Would need psutil for accurate CPU measurement
                patterns_per_second=patterns_per_second,
                data_size=data_size,
            )

            self.performance_history.append(metrics)

            # Keep only last 100 performance records
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available

    def _calculate_memory_metrics(self, start_memory: float) -> Dict[str, float]:
        """Calculate memory usage metrics"""
        try:
            current_memory = self._get_memory_usage()
            return {
                "start_mb": start_memory,
                "current_mb": current_memory,
                "peak_mb": max(start_memory, current_memory),
                "delta_mb": current_memory - start_memory,
            }
        except Exception:
            return {"start_mb": 0, "current_mb": 0, "peak_mb": 0, "delta_mb": 0}

    def _optimize_memory(self) -> None:
        """Optimize memory usage before analysis"""
        try:
            if self.memory_optimization:
                # Force garbage collection
                gc.collect()

                # Clear validation cache if it exists
                if hasattr(self, "validation_engine") and hasattr(
                    self.validation_engine, "reliability_cache"
                ):
                    if len(self.validation_engine.reliability_cache) > 1000:
                        self.validation_engine.reliability_cache.clear()
                        logger.debug("Cleared validation cache for memory optimization")

        except Exception as e:
            logger.error(f"Error in memory optimization: {e}")

    def _cleanup_after_analysis(self) -> None:
        """Cleanup after analysis to free memory"""
        try:
            if self.memory_optimization:
                # Force garbage collection
                gc.collect()

                # Check memory usage and cleanup if needed
                current_memory = self._get_memory_usage()
                if current_memory > self._memory_threshold_mb:
                    logger.info(
                        f"Memory usage ({current_memory:.1f}MB) exceeds threshold, performing cleanup"
                    )

                    # Limit performance history
                    if len(self.performance_history) > 50:
                        self.performance_history = self.performance_history[-50:]

                    # Clear timeframe manager cache
                    if hasattr(self.timeframe_manager, "timeframe_data"):
                        for (
                            tf_name,
                            tf_data,
                        ) in self.timeframe_manager.timeframe_data.items():
                            if hasattr(tf_data, "data") and len(tf_data.data) > 1000:
                                tf_data.data = tf_data.data.tail(500).copy()

                        logger.debug(
                            "Cleaned up timeframe data for memory optimization"
                        )

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    # Signal generation helpers

    def _generate_order_block_signal(
        self, order_block: OrderBlock, data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal from Order Block"""
        try:
            current_price = data["close"].iloc[-1]

            # Check if price is near order block
            ob_distance = min(
                abs(current_price - order_block.high_price),
                abs(current_price - order_block.low_price),
            )

            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[-1]
            )

            if ob_distance <= atr * 0.5:  # Within 0.5 ATR of order block
                return {
                    "type": "order_block",
                    "direction": order_block.direction.value,
                    "strength": min(1.0, (atr * 0.5 - ob_distance) / (atr * 0.5)),
                    "entry_zone": {
                        "high": order_block.high_price,
                        "low": order_block.low_price,
                    },
                    "timestamp": pd.Timestamp.now(),
                }

        except Exception as e:
            logger.error(f"Error generating order block signal: {e}")

        return None

    def _generate_fvg_signal(
        self, fvg: FairValueGap, data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal from Fair Value Gap"""
        try:
            current_price = data["close"].iloc[-1]

            # Check if price is approaching FVG
            if fvg.direction.value == "bullish" and current_price <= fvg.gap_high:
                return {
                    "type": "fair_value_gap",
                    "direction": "bullish",
                    "strength": 0.7,
                    "gap_zone": {"high": fvg.gap_high, "low": fvg.gap_low},
                    "timestamp": pd.Timestamp.now(),
                }
            elif fvg.direction.value == "bearish" and current_price >= fvg.gap_low:
                return {
                    "type": "fair_value_gap",
                    "direction": "bearish",
                    "strength": 0.7,
                    "gap_zone": {"high": fvg.gap_high, "low": fvg.gap_low},
                    "timestamp": pd.Timestamp.now(),
                }

        except Exception as e:
            logger.error(f"Error generating FVG signal: {e}")

        return None

    def _generate_structure_signal(
        self, structure: MarketStructure, data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal from Market Structure break"""
        try:
            return {
                "type": "market_structure",
                "pattern_type": structure.pattern_type,
                "direction": structure.direction.value,
                "strength": min(1.0, structure.confidence_score),
                "break_price": structure.break_price,
                "timestamp": pd.Timestamp.now(),
            }

        except Exception as e:
            logger.error(f"Error generating structure signal: {e}")

        return None

    def _filter_and_rank_signals(
        self, signals: List[Dict[str, Any]], data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Filter and rank signals by strength and relevance"""
        try:
            # Filter out weak signals
            filtered_signals = [s for s in signals if s.get("strength", 0) > 0.5]

            # Sort by strength (descending)
            filtered_signals.sort(key=lambda x: x.get("strength", 0), reverse=True)

            # Limit to top 5 signals
            return filtered_signals[:5]

        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals
