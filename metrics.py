import time
import tracemalloc
import gc
from dataclasses import dataclass
from typing import Callable, Any, Dict

@dataclass
class BenchmarkResult:
    """Structure to hold benchmark results."""
    execution_time_sec: float
    peak_memory_kb: float
    
    def __str__(self):
        return (f"  Time: {self.execution_time_sec:.6f} sec\n"
                f"  Peak Memory: {self.peak_memory_kb:.2f} KB")

def measure_latency(func: Callable, *args, warmups: int = 10, iterations: int = 100) -> float:
    """
    Measure average execution time of a function.
    
    Args:
        func: function to benchmark
        *args: arguments to pass to the function
        warmups: number of warmup iterations (ignored in timing)
        iterations: number of measured iterations
        
    Returns:
        Average time per iteration in seconds.
    """
    # Warmup
    for _ in range(warmups):
        func(*args)
        
    # Timing
    start_time = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    return total_time / iterations

def measure_peak_memory(func: Callable, *args, iterations: int = 10) -> float:
    """
    Measure peak memory increase during execution.
    
    Args:
        func: function to benchmark
        *args: arguments to pass to the function
        iterations: number of iterations to run trace over
        
    Returns:
        Peak memory usage in KB.
    """
    gc.collect()
    tracemalloc.start()
    
    try:
        # Run multiple times to capture peak allocation
        for _ in range(iterations):
            func(*args)
            
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        
    return peak / 1024.0  # Convert bytes to KB

def run_benchmark(name: str, func: Callable, *args, warmups: int = 10, iterations: int = 50) -> BenchmarkResult:
    """
    Run full benchmark (latency + memory) for a specific function.
    """
    print(f"Benchmarking {name}...")
    
    latency = measure_latency(func, *args, warmups=warmups, iterations=iterations)
    peak_mem = measure_peak_memory(func, *args, iterations=iterations)
    
    return BenchmarkResult(latency, peak_mem)
