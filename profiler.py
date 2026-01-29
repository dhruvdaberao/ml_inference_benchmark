import time
import tracemalloc
import gc
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable, Any, Dict

@dataclass
class ProfileResult:
    latency_sec: float
    peak_memory_kb: float
    output_summary: Dict[str, float]

class ExecutionProfiler:
    """
    Handles benchmarking of execution engines.
    """
    
    def __init__(self, warmups: int = 10, iterations: int = 50):
        self.warmups = warmups
        self.iterations = iterations
        
    def profile(self, func: Callable, *args) -> ProfileResult:
        """
        Run the full profile (latency + memory) on a function.
        """
        # 1. Warmup
        for _ in range(self.warmups):
            func(*args)
            
        # 2. Measure Latency
        start_time = time.perf_counter()
        for _ in range(self.iterations):
            last_output = func(*args)
        end_time = time.perf_counter()
        avg_latency = (end_time - start_time) / self.iterations
        
        # 3. Measure Memory
        gc.collect()
        tracemalloc.start()
        try:
            # Run a few times to trigger allocations
            for _ in range(self.iterations):
                func(*args)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            
        peak_kb = peak / 1024.0
        
        # 4. Summarize Output (of the last run)
        # Assuming output is numpy array
        output_stats = {
            "min": float(last_output.min()),
            "max": float(last_output.max()),
            "mean": float(last_output.mean())
        }
        
        return ProfileResult(
            latency_sec=avg_latency,
            peak_memory_kb=peak_kb,
            output_summary=output_stats
        )
