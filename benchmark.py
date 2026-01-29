import numpy as np
import sys
import os

# Add parent directory to path to handle imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_inference_benchmark.models import MLPModel
from ml_inference_benchmark.runner import BaselineRunner, OptimizedRunner
from ml_inference_benchmark.metrics import run_benchmark

def main():
    print("="*60)
    print("AI Compiler Inference Benchmark (NumPy Only)")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 64
    INPUT_DIM = 1024
    HIDDEN_DIM = 4096  # Large enough to show allocations matter
    OUTPUT_DIM = 1024
    SEED = 42
    
    print(f"Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Input: {INPUT_DIM}")
    print(f"  Hidden: {HIDDEN_DIM}")
    print(f"  Output: {OUTPUT_DIM}")
    print("-"*60)

    # Data Generation
    print("Generating fixed input data...")
    rng = np.random.RandomState(SEED)
    input_data = rng.randn(BATCH_SIZE, INPUT_DIM).astype(np.float32)
    
    # Model Initialization
    print("Initializing weights...")
    model = MLPModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, seed=SEED)
    weights = model.get_weights()
    
    # Runners
    baseline = BaselineRunner(weights)
    optimized = OptimizedRunner(weights, max_batch_size=BATCH_SIZE, 
                              input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
    
    # 1. Correctness Check
    print("\nVerifying numerical correctness...")
    res_base = baseline.forward(input_data)
    res_opt = optimized.forward(input_data)
    
    diff = np.abs(res_base - res_opt).max()
    print(f"  Max absolute difference: {diff:.2e}")
    if diff > 1e-5:
        print("  [FAIL] Mismatch detected! Optimization is flawed.")
        sys.exit(1)
    print("  [OK] Outputs match.")
    
    # 2. Benchmarking
    print("\nStarting Benchmarks (Warming up & Measuring)...")
    
    # Baseline
    res_b = run_benchmark("Baseline Execution", baseline.forward, input_data, iterations=50)
    
    # Optimized
    res_o = run_benchmark("Optimized Execution", optimized.forward, input_data, iterations=50)
    
    # 3. Summary Report
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print("Baseline:")
    print(str(res_b))
    
    print("\nOptimized (Compiler-Style):")
    print(str(res_o))
    
    # Calculate Speedup / Savings
    speedup = res_b.execution_time_sec / res_o.execution_time_sec
    mem_saved = res_b.peak_memory_kb - res_o.peak_memory_kb
    mem_reduction = (mem_saved / res_b.peak_memory_kb) * 100 if res_b.peak_memory_kb > 0 else 0
    
    print("-" * 60)
    print(f"* Performance Win:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Memory:  {mem_reduction:.1f}% reduction in peak allocation")
    print("="*60)

if __name__ == "__main__":
    main()
