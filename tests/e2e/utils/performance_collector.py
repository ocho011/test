"""
Performance metrics collection for E2E testing.

Collects latency, throughput, and resource metrics during test execution.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class LatencyMetric:
    """Latency measurement between two points."""
    start_point: str
    end_point: str
    latency_ms: float
    timestamp: datetime


@dataclass
class ThroughputMetric:
    """Throughput measurement for a component."""
    component: str
    items_processed: int
    duration_seconds: float
    items_per_second: float
    timestamp: datetime


@dataclass
class ResourceMetric:
    """Resource usage snapshot."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    timestamp: datetime


@dataclass
class PerformanceReport:
    """Complete performance test report."""
    latencies: List[LatencyMetric] = field(default_factory=list)
    throughputs: List[ThroughputMetric] = field(default_factory=list)
    resources: List[ResourceMetric] = field(default_factory=list)
    
    def get_avg_latency(self, start_point: str, end_point: str) -> Optional[float]:
        """Get average latency between two points."""
        latencies = [
            m.latency_ms for m in self.latencies
            if m.start_point == start_point and m.end_point == end_point
        ]
        return sum(latencies) / len(latencies) if latencies else None
    
    def get_max_latency(self, start_point: str, end_point: str) -> Optional[float]:
        """Get maximum latency between two points."""
        latencies = [
            m.latency_ms for m in self.latencies
            if m.start_point == start_point and m.end_point == end_point
        ]
        return max(latencies) if latencies else None
    
    def get_avg_throughput(self, component: str) -> Optional[float]:
        """Get average throughput for a component."""
        throughputs = [
            m.items_per_second for m in self.throughputs
            if m.component == component
        ]
        return sum(throughputs) / len(throughputs) if throughputs else None
    
    def get_peak_memory_mb(self) -> Optional[float]:
        """Get peak memory usage in MB."""
        if not self.resources:
            return None
        return max(m.memory_mb for m in self.resources)
    
    def get_avg_cpu_percent(self) -> Optional[float]:
        """Get average CPU usage percentage."""
        if not self.resources:
            return None
        return sum(m.cpu_percent for m in self.resources) / len(self.resources)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "latencies": {
                "count": len(self.latencies),
                "measurements": [
                    {
                        "flow": f"{m.start_point} → {m.end_point}",
                        "avg_ms": self.get_avg_latency(m.start_point, m.end_point),
                        "max_ms": self.get_max_latency(m.start_point, m.end_point)
                    }
                    for m in self.latencies
                ]
            },
            "throughputs": {
                "count": len(self.throughputs),
                "measurements": [
                    {
                        "component": m.component,
                        "avg_items_per_sec": self.get_avg_throughput(m.component)
                    }
                    for m in self.throughputs
                ]
            },
            "resources": {
                "peak_memory_mb": self.get_peak_memory_mb(),
                "avg_cpu_percent": self.get_avg_cpu_percent(),
                "samples": len(self.resources)
            }
        }


class PerformanceCollector:
    """
    Collects performance metrics during E2E tests.
    
    Features:
    - Latency measurement between checkpoints
    - Throughput calculation for components
    - Resource usage monitoring
    - Automated reporting
    """

    def __init__(self):
        """Initialize performance collector."""
        self.report = PerformanceReport()
        self._checkpoints: Dict[str, float] = {}
        self._throughput_counters: Dict[str, List[int]] = defaultdict(list)
        self._throughput_start_times: Dict[str, float] = {}
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self.process = psutil.Process()

    def checkpoint(self, name: str) -> None:
        """
        Record a timing checkpoint.
        
        Args:
            name: Checkpoint name
        """
        self._checkpoints[name] = time.perf_counter()

    def record_latency(self, start_checkpoint: str, end_checkpoint: str) -> Optional[float]:
        """
        Record latency between two checkpoints.
        
        Args:
            start_checkpoint: Starting checkpoint name
            end_checkpoint: Ending checkpoint name
            
        Returns:
            Latency in milliseconds, or None if checkpoints not found
        """
        if start_checkpoint not in self._checkpoints:
            return None
        if end_checkpoint not in self._checkpoints:
            return None
        
        start_time = self._checkpoints[start_checkpoint]
        end_time = self._checkpoints[end_checkpoint]
        latency_ms = (end_time - start_time) * 1000.0
        
        metric = LatencyMetric(
            start_point=start_checkpoint,
            end_point=end_checkpoint,
            latency_ms=latency_ms,
            timestamp=datetime.now()
        )
        
        self.report.latencies.append(metric)
        return latency_ms

    def start_throughput(self, component: str) -> None:
        """
        Start throughput measurement for a component.
        
        Args:
            component: Component name
        """
        self._throughput_start_times[component] = time.perf_counter()
        self._throughput_counters[component] = []

    def increment_throughput(self, component: str, count: int = 1) -> None:
        """
        Increment throughput counter for a component.
        
        Args:
            component: Component name
            count: Number of items processed
        """
        if component in self._throughput_counters:
            self._throughput_counters[component].append(count)

    def end_throughput(self, component: str) -> Optional[float]:
        """
        End throughput measurement and calculate rate.
        
        Args:
            component: Component name
            
        Returns:
            Items per second, or None if not started
        """
        if component not in self._throughput_start_times:
            return None
        
        start_time = self._throughput_start_times[component]
        duration = time.perf_counter() - start_time
        items_processed = sum(self._throughput_counters.get(component, []))
        
        if duration > 0:
            items_per_second = items_processed / duration
        else:
            items_per_second = 0.0
        
        metric = ThroughputMetric(
            component=component,
            items_processed=items_processed,
            duration_seconds=duration,
            items_per_second=items_per_second,
            timestamp=datetime.now()
        )
        
        self.report.throughputs.append(metric)
        
        # Clean up
        del self._throughput_start_times[component]
        del self._throughput_counters[component]
        
        return items_per_second

    async def start_resource_monitoring(self, interval_seconds: float = 1.0) -> None:
        """
        Start monitoring resource usage.
        
        Args:
            interval_seconds: Sampling interval
        """
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._resource_monitor_task = asyncio.create_task(
            self._monitor_resources(interval_seconds)
        )

    async def stop_resource_monitoring(self) -> None:
        """Stop monitoring resource usage."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
            try:
                await self._resource_monitor_task
            except asyncio.CancelledError:
                pass
            self._resource_monitor_task = None

    async def _monitor_resources(self, interval: float) -> None:
        """
        Monitor resource usage periodically.
        
        Args:
            interval: Sampling interval in seconds
        """
        while self._is_monitoring:
            try:
                # Get CPU and memory usage
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                memory_percent = self.process.memory_percent()
                
                metric = ResourceMetric(
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    timestamp=datetime.now()
                )
                
                self.report.resources.append(metric)
                
                await asyncio.sleep(interval)
                
            except Exception:
                # Ignore errors during monitoring
                pass

    def assert_latency_within(
        self,
        start_checkpoint: str,
        end_checkpoint: str,
        max_latency_ms: float
    ) -> None:
        """
        Assert that latency is within acceptable range.
        
        Args:
            start_checkpoint: Starting checkpoint
            end_checkpoint: Ending checkpoint
            max_latency_ms: Maximum acceptable latency in milliseconds
            
        Raises:
            AssertionError: If latency exceeds limit
        """
        latencies = [
            m.latency_ms for m in self.report.latencies
            if m.start_point == start_checkpoint and m.end_point == end_checkpoint
        ]
        
        if not latencies:
            raise AssertionError(
                f"No latency measurements found for {start_checkpoint} → {end_checkpoint}"
            )
        
        max_measured = max(latencies)
        avg_measured = sum(latencies) / len(latencies)
        
        if max_measured > max_latency_ms:
            raise AssertionError(
                f"Latency {start_checkpoint} → {end_checkpoint} exceeded limit: "
                f"{max_measured:.2f}ms > {max_latency_ms:.2f}ms "
                f"(avg: {avg_measured:.2f}ms)"
            )

    def assert_throughput_above(
        self,
        component: str,
        min_items_per_second: float
    ) -> None:
        """
        Assert that throughput meets minimum requirement.
        
        Args:
            component: Component name
            min_items_per_second: Minimum acceptable throughput
            
        Raises:
            AssertionError: If throughput below limit
        """
        throughputs = [
            m.items_per_second for m in self.report.throughputs
            if m.component == component
        ]
        
        if not throughputs:
            raise AssertionError(
                f"No throughput measurements found for {component}"
            )
        
        avg_throughput = sum(throughputs) / len(throughputs)
        
        if avg_throughput < min_items_per_second:
            raise AssertionError(
                f"Throughput for {component} below minimum: "
                f"{avg_throughput:.2f} < {min_items_per_second:.2f} items/sec"
            )

    def clear(self) -> None:
        """Clear all collected metrics."""
        self.report = PerformanceReport()
        self._checkpoints.clear()
        self._throughput_counters.clear()
        self._throughput_start_times.clear()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerformanceCollector("
            f"latencies={len(self.report.latencies)}, "
            f"throughputs={len(self.report.throughputs)}, "
            f"resources={len(self.report.resources)})"
        )
