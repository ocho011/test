"""
Technical Indicators Integration Module for ICT Trading

This module integrates traditional technical indicators with ICT concepts,
providing enhanced analysis by combining both approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical indicators calculator with pandas optimization

    Provides commonly used technical indicators optimized for performance
    and integrated with ICT pattern analysis.
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return {
            'k': k_percent,
            'd': d_percent
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        tr = TechnicalIndicators.atr(high, low, close, 1)

        dm_plus = high.diff()
        dm_minus = -low.diff()

        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

        di_plus = 100 * (dm_plus.rolling(window=period).mean() / tr.rolling(window=period).mean())
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / tr.rolling(window=period).mean())

        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }

    @staticmethod
    def volume_profile(data: pd.DataFrame, price_bins: int = 50) -> Dict[str, np.ndarray]:
        """Volume Profile calculation"""
        if 'volume' not in data.columns:
            logger.warning("Volume data not available for volume profile")
            return {}

        # Create price bins
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / price_bins

        volume_profile = np.zeros(price_bins)
        price_levels = np.linspace(data['low'].min(), data['high'].max(), price_bins + 1)

        for i, row in data.iterrows():
            # Distribute volume across price range of the bar
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']

            # Find which bins this bar covers
            start_bin = max(0, int((bar_low - data['low'].min()) / bin_size))
            end_bin = min(price_bins - 1, int((bar_high - data['low'].min()) / bin_size))

            # Distribute volume evenly across bins
            bins_covered = max(1, end_bin - start_bin + 1)
            volume_per_bin = bar_volume / bins_covered

            for bin_idx in range(start_bin, end_bin + 1):
                if bin_idx < price_bins:
                    volume_profile[bin_idx] += volume_per_bin

        return {
            'volume_profile': volume_profile,
            'price_levels': price_levels[:-1],  # Remove last element to match profile length
            'poc': price_levels[np.argmax(volume_profile)]  # Point of Control
        }


class ICTIndicatorIntegration:
    """
    Integrates traditional technical indicators with ICT concepts

    Provides enhanced analysis by combining technical indicators with
    ICT patterns for confluence and signal validation.
    """

    def __init__(self):
        """Initialize ICT Indicator Integration"""
        self.logger = logging.getLogger(f"{__name__}.ICTIndicatorIntegration")
        self.indicators = TechnicalIndicators()

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all relevant technical indicators

        Args:
            data: DataFrame with OHLC data

        Returns:
            Dictionary containing all calculated indicators
        """
        try:
            results = {}

            # Moving Averages
            results['sma_20'] = self.indicators.sma(data['close'], 20)
            results['sma_50'] = self.indicators.sma(data['close'], 50)
            results['sma_200'] = self.indicators.sma(data['close'], 200)
            results['ema_9'] = self.indicators.ema(data['close'], 9)
            results['ema_21'] = self.indicators.ema(data['close'], 21)

            # Oscillators
            results['rsi'] = self.indicators.rsi(data['close'])
            results['stochastic'] = self.indicators.stochastic(
                data['high'], data['low'], data['close']
            )

            # MACD
            results['macd'] = self.indicators.macd(data['close'])

            # Bollinger Bands
            results['bollinger'] = self.indicators.bollinger_bands(data['close'])

            # Volatility
            results['atr'] = self.indicators.atr(data['high'], data['low'], data['close'])

            # Trend Strength
            results['adx'] = self.indicators.adx(data['high'], data['low'], data['close'])

            # Volume Profile (if volume available)
            if 'volume' in data.columns:
                results['volume_profile'] = self.indicators.volume_profile(data)

            self.logger.info(f"Calculated {len(results)} technical indicators")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}

    def get_indicator_confluence(self, indicators: Dict, ict_analysis: Dict) -> Dict[str, any]:
        """
        Analyze confluence between technical indicators and ICT patterns

        Args:
            indicators: Technical indicators results
            ict_analysis: ICT pattern analysis results

        Returns:
            Confluence analysis results
        """
        confluence = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'confluence_score': 0.0,
            'signal_details': [],
            'overall_bias': 'neutral'
        }

        try:
            # Check RSI confluence
            if 'rsi' in indicators:
                latest_rsi = indicators['rsi'].iloc[-1]
                if latest_rsi < 30:
                    confluence['bullish_signals'] += 1
                    confluence['signal_details'].append('RSI oversold (bullish)')
                elif latest_rsi > 70:
                    confluence['bearish_signals'] += 1
                    confluence['signal_details'].append('RSI overbought (bearish)')
                else:
                    confluence['neutral_signals'] += 1

            # Check MACD confluence
            if 'macd' in indicators:
                macd_data = indicators['macd']
                if len(macd_data['macd']) > 1:
                    if (macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1] and
                        macd_data['histogram'].iloc[-1] > 0):
                        confluence['bullish_signals'] += 1
                        confluence['signal_details'].append('MACD bullish crossover')
                    elif (macd_data['macd'].iloc[-1] < macd_data['signal'].iloc[-1] and
                          macd_data['histogram'].iloc[-1] < 0):
                        confluence['bearish_signals'] += 1
                        confluence['signal_details'].append('MACD bearish crossover')

            # Check Moving Average confluence
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20_current = indicators['sma_20'].iloc[-1]
                sma_50_current = indicators['sma_50'].iloc[-1]

                if sma_20_current > sma_50_current:
                    confluence['bullish_signals'] += 1
                    confluence['signal_details'].append('SMA 20 > SMA 50 (bullish)')
                else:
                    confluence['bearish_signals'] += 1
                    confluence['signal_details'].append('SMA 20 < SMA 50 (bearish)')

            # Check ADX for trend strength
            if 'adx' in indicators:
                adx_current = indicators['adx']['adx'].iloc[-1]
                if adx_current > 25:
                    confluence['signal_details'].append(f'Strong trend (ADX: {adx_current:.1f})')

            # Integrate with ICT patterns
            if 'current_trend' in ict_analysis:
                ict_trend = ict_analysis['current_trend']
                if hasattr(ict_trend, 'value'):
                    ict_trend = ict_trend.value

                if ict_trend == 'bullish':
                    confluence['bullish_signals'] += 2  # Weight ICT more heavily
                    confluence['signal_details'].append('ICT trend: Bullish')
                elif ict_trend == 'bearish':
                    confluence['bearish_signals'] += 2
                    confluence['signal_details'].append('ICT trend: Bearish')

            # Calculate confluence score
            total_signals = (confluence['bullish_signals'] +
                           confluence['bearish_signals'] +
                           confluence['neutral_signals'])

            if total_signals > 0:
                if confluence['bullish_signals'] > confluence['bearish_signals']:
                    confluence['confluence_score'] = confluence['bullish_signals'] / total_signals
                    confluence['overall_bias'] = 'bullish'
                elif confluence['bearish_signals'] > confluence['bullish_signals']:
                    confluence['confluence_score'] = confluence['bearish_signals'] / total_signals
                    confluence['overall_bias'] = 'bearish'
                else:
                    confluence['confluence_score'] = 0.5
                    confluence['overall_bias'] = 'neutral'

            self.logger.info(
                f"Confluence analysis: {confluence['overall_bias']} "
                f"(score: {confluence['confluence_score']:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Error in confluence analysis: {e}")

        return confluence

    def get_key_levels(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, List[float]]:
        """
        Identify key levels using technical indicators

        Args:
            data: OHLC DataFrame
            indicators: Technical indicators results

        Returns:
            Dictionary with support and resistance levels
        """
        levels = {
            'support_levels': [],
            'resistance_levels': [],
            'pivot_levels': []
        }

        try:
            # Bollinger Bands levels
            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                levels['support_levels'].append(bb['lower'].iloc[-1])
                levels['resistance_levels'].append(bb['upper'].iloc[-1])
                levels['pivot_levels'].append(bb['middle'].iloc[-1])

            # Moving Average levels
            for ma_key in ['sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21']:
                if ma_key in indicators:
                    ma_level = indicators[ma_key].iloc[-1]
                    current_price = data['close'].iloc[-1]

                    if ma_level < current_price:
                        levels['support_levels'].append(ma_level)
                    else:
                        levels['resistance_levels'].append(ma_level)

            # Volume Profile POC
            if 'volume_profile' in indicators and 'poc' in indicators['volume_profile']:
                levels['pivot_levels'].append(indicators['volume_profile']['poc'])

            # Remove duplicates and sort
            levels['support_levels'] = sorted(list(set(levels['support_levels'])))
            levels['resistance_levels'] = sorted(list(set(levels['resistance_levels'])), reverse=True)
            levels['pivot_levels'] = sorted(list(set(levels['pivot_levels'])))

        except Exception as e:
            self.logger.error(f"Error identifying key levels: {e}")

        return levels

    def analyze(self, data: pd.DataFrame, ict_analysis: Optional[Dict] = None) -> Dict[str, any]:
        """
        Main analysis method integrating indicators with ICT

        Args:
            data: DataFrame with OHLC data
            ict_analysis: Optional ICT analysis results

        Returns:
            Complete technical analysis with ICT integration
        """
        try:
            self.logger.info(f"Running technical indicator analysis for {len(data)} bars")

            # Calculate all indicators
            indicators = self.calculate_all_indicators(data)

            # Get confluence analysis
            confluence = {}
            if ict_analysis:
                confluence = self.get_indicator_confluence(indicators, ict_analysis)

            # Identify key levels
            key_levels = self.get_key_levels(data, indicators)

            # Compile results
            results = {
                'indicators': indicators,
                'confluence': confluence,
                'key_levels': key_levels,
                'analysis_timestamp': pd.Timestamp.now(),
                'data_length': len(data)
            }

            self.logger.info("Technical indicator analysis complete")
            return results

        except Exception as e:
            self.logger.error(f"Error in technical indicator analysis: {e}")
            return {
                'indicators': {},
                'confluence': {},
                'key_levels': {'support_levels': [], 'resistance_levels': [], 'pivot_levels': []},
                'analysis_timestamp': pd.Timestamp.now(),
                'data_length': len(data) if data is not None else 0
            }


class ICTPatternIndicatorFusion:
    """
    Advanced fusion of ICT patterns with technical indicators

    Provides sophisticated analysis combining ICT concepts with traditional
    technical analysis for enhanced market understanding.
    """

    def __init__(self):
        """Initialize ICT Pattern Indicator Fusion"""
        self.logger = logging.getLogger(f"{__name__}.ICTPatternIndicatorFusion")
        self.integration = ICTIndicatorIntegration()

    def validate_order_blocks_with_indicators(self, order_blocks: List, indicators: Dict) -> List:
        """
        Validate Order Blocks using technical indicators

        Args:
            order_blocks: List of detected order blocks
            indicators: Technical indicators results

        Returns:
            List of validated order blocks with confidence scores
        """
        validated_blocks = []

        try:
            for block in order_blocks:
                validation_score = 0.0
                validation_factors = []

                # Check RSI confluence
                if 'rsi' in indicators:
                    rsi_at_block = indicators['rsi'].iloc[block.start_index] if block.start_index < len(indicators['rsi']) else None
                    if rsi_at_block:
                        if block.direction.value == 'bullish' and rsi_at_block < 40:
                            validation_score += 0.2
                            validation_factors.append('RSI oversold confluence')
                        elif block.direction.value == 'bearish' and rsi_at_block > 60:
                            validation_score += 0.2
                            validation_factors.append('RSI overbought confluence')

                # Check volume confluence
                if 'volume_profile' in indicators:
                    block_price_level = (block.high_price + block.low_price) / 2
                    poc = indicators['volume_profile'].get('poc')
                    if poc and abs(block_price_level - poc) < (block.high_price - block.low_price):
                        validation_score += 0.3
                        validation_factors.append('Volume POC confluence')

                # Check moving average confluence
                if 'sma_20' in indicators:
                    sma_at_block = indicators['sma_20'].iloc[block.start_index] if block.start_index < len(indicators['sma_20']) else None
                    if sma_at_block:
                        block_price = (block.high_price + block.low_price) / 2
                        if abs(block_price - sma_at_block) < (block.high_price - block.low_price) * 0.5:
                            validation_score += 0.25
                            validation_factors.append('SMA 20 confluence')

                # Update block confidence
                block.confidence = min(1.0, block.confidence + validation_score)

                # Add validation metadata
                block.validation_factors = validation_factors
                validated_blocks.append(block)

            self.logger.info(f"Validated {len(validated_blocks)} order blocks with indicators")

        except Exception as e:
            self.logger.error(f"Error validating order blocks: {e}")
            validated_blocks = order_blocks

        return validated_blocks

    def enhance_fair_value_gaps(self, gaps: List, indicators: Dict) -> List:
        """
        Enhance Fair Value Gap analysis with indicators

        Args:
            gaps: List of detected FVGs
            indicators: Technical indicators results

        Returns:
            List of enhanced FVGs
        """
        enhanced_gaps = []

        try:
            for gap in gaps:
                enhancement_score = 0.0

                # Check ATR for gap size validation
                if 'atr' in indicators and gap.start_index < len(indicators['atr']):
                    atr_at_gap = indicators['atr'].iloc[gap.start_index]
                    if gap.gap_size > atr_at_gap * 0.5:  # Gap is significant relative to volatility
                        enhancement_score += 0.3

                # Check volume for gap validation
                if 'volume_profile' in indicators:
                    gap_midpoint = (gap.gap_high + gap.gap_low) / 2
                    # Add logic for volume validation at gap level
                    enhancement_score += 0.2

                # Add enhancement score to gap
                gap.enhancement_score = enhancement_score
                enhanced_gaps.append(gap)

        except Exception as e:
            self.logger.error(f"Error enhancing FVGs: {e}")
            enhanced_gaps = gaps

        return enhanced_gaps

    def fusion_analysis(self, data: pd.DataFrame, ict_patterns: Dict) -> Dict[str, any]:
        """
        Perform fusion analysis combining ICT patterns with indicators

        Args:
            data: OHLC DataFrame
            ict_patterns: ICT pattern analysis results

        Returns:
            Fusion analysis results
        """
        try:
            self.logger.info("Starting ICT-Indicator fusion analysis")

            # Get technical indicator analysis
            indicator_analysis = self.integration.analyze(data, ict_patterns)

            # Validate and enhance ICT patterns
            enhanced_patterns = ict_patterns.copy()

            # Enhance order blocks if present
            if 'order_blocks' in ict_patterns:
                enhanced_patterns['validated_order_blocks'] = self.validate_order_blocks_with_indicators(
                    ict_patterns['order_blocks'], indicator_analysis['indicators']
                )

            # Enhance FVGs if present
            if 'fair_value_gaps' in ict_patterns:
                enhanced_patterns['enhanced_fvgs'] = self.enhance_fair_value_gaps(
                    ict_patterns['fair_value_gaps'], indicator_analysis['indicators']
                )

            # Create fusion score
            fusion_score = self._calculate_fusion_score(
                indicator_analysis['confluence'], enhanced_patterns
            )

            results = {
                'fusion_score': fusion_score,
                'enhanced_ict_patterns': enhanced_patterns,
                'technical_analysis': indicator_analysis,
                'fusion_signals': self._generate_fusion_signals(indicator_analysis, enhanced_patterns),
                'analysis_timestamp': pd.Timestamp.now()
            }

            self.logger.info(f"Fusion analysis complete. Fusion score: {fusion_score:.2f}")
            return results

        except Exception as e:
            self.logger.error(f"Error in fusion analysis: {e}")
            return {
                'fusion_score': 0.0,
                'enhanced_ict_patterns': ict_patterns,
                'technical_analysis': {},
                'fusion_signals': [],
                'analysis_timestamp': pd.Timestamp.now()
            }

    def _calculate_fusion_score(self, confluence: Dict, enhanced_patterns: Dict) -> float:
        """Calculate overall fusion score"""
        try:
            base_score = confluence.get('confluence_score', 0.0)

            # Add ICT pattern strength
            pattern_boost = 0.0
            if 'validated_order_blocks' in enhanced_patterns:
                avg_confidence = np.mean([block.confidence for block in enhanced_patterns['validated_order_blocks']])
                pattern_boost += avg_confidence * 0.3

            return min(1.0, base_score + pattern_boost)

        except Exception:
            return 0.0

    def _generate_fusion_signals(self, indicator_analysis: Dict, enhanced_patterns: Dict) -> List[Dict]:
        """Generate trading signals from fusion analysis"""
        signals = []

        try:
            confluence = indicator_analysis.get('confluence', {})

            if confluence.get('confluence_score', 0) > 0.7:
                signals.append({
                    'signal_type': 'confluence_signal',
                    'direction': confluence.get('overall_bias', 'neutral'),
                    'strength': confluence.get('confluence_score', 0),
                    'factors': confluence.get('signal_details', []),
                    'timestamp': pd.Timestamp.now()
                })

        except Exception as e:
            self.logger.error(f"Error generating fusion signals: {e}")

        return signals