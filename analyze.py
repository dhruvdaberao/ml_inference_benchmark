import argparse
import sys
import datetime
from typing import Dict, Any, List
from dataclasses import asdict
import numpy as np

from models import MLPModel
from profiler import ExecutionProfiler
from utils import parse_input_string, tile_input_data, format_summary, save_report

# Import Engines
from engines.naive import NaiveExecutionEngine
from engines.optimized import OptimizedExecutionEngine

# Configuration Constants
INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1024

def run_analysis(input_str: str, batch_size: int, mode: str) -> Dict[str, Any]:
    """
    Core analysis logic shared between CLI and Web.
    """
    # 1. Prepare Data
    try:
        raw_input = parse_input_string(input_str)
        input_data = tile_input_data(raw_input, batch_size, INPUT_DIM)
    except Exception as e:
        raise ValueError(f"Error preparing input: {e}")

    # 2. Initialize Model
    model = MLPModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    weights = model.get_weights()
    
    config = {
        'max_batch_size': batch_size,
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': OUTPUT_DIM
    }
    
    # 3. Initialize Engines
    engines = {}
    if mode in ['baseline', 'both']:
        engines['baseline'] = NaiveExecutionEngine(weights, config)
    if mode in ['optimized', 'both']:
        engines['optimized'] = OptimizedExecutionEngine(weights, config)
        
    profiler = ExecutionProfiler(warmups=5, iterations=20)
    results = {}
    
    # 4. Execute & Profile
    for name, engine in engines.items():
        # print(f"Executing {name}...") # Silence prints for web usage, or keep for logs?
        # Keeping minimal logging could be fine or redirected. For now, we assume this function returns data.
        res = profiler.profile(engine.forward, input_data)
        results[name] = res
        
    # 5. Comparison & Validation
    report_data = {
        "timestamp": str(datetime.datetime.now()),
        "mode": mode,
        "config": {"batch_size": batch_size},
        "input_preview": input_str[:50] + "..."
    }
    
    for name, res in results.items():
        report_data[name] = asdict(res)
        
    if mode == 'both':
        base = results['baseline']
        opt = results['optimized']
        
        speedup = base.latency_sec / opt.latency_sec if opt.latency_sec > 0 else 0
        mem_saved = base.peak_memory_kb - opt.peak_memory_kb
        mem_saved_pct = (mem_saved / base.peak_memory_kb * 100) if base.peak_memory_kb > 0 else 0
        
        # Correctness Check
        out_base = engines['baseline'].forward(input_data)
        out_opt = engines['optimized'].forward(input_data)
        max_diff = np.abs(out_base - out_opt).max()
        correctness = max_diff < 1e-4
        
        report_data['comparison'] = {
            "speedup_x": float(speedup),
            "memory_savings_percent": float(mem_saved_pct),
            "correctness": bool(correctness),
            "max_diff": float(max_diff)
        }
        
    return report_data


def main():
    parser = argparse.ArgumentParser(description="ML Execution & Optimization Analyzer")
    parser.add_argument("--input", type=str, required=True, help="Comma-separated input values (e.g., '1.0,0.5,-0.2')")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for execution")
    parser.add_argument("--mode", type=str, choices=['baseline', 'optimized', 'both'], default='both', help="Execution mode")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ML EXECUTION ANALYZER")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Batch Size: {args.batch}")
    print(f"Model: MLP ({INPUT_DIM} -> {HIDDEN_DIM} -> {OUTPUT_DIM})")
    
    try:
        report_data = run_analysis(args.input, args.batch, args.mode)
    except Exception as e:
        print(f"Analysis Failed: {e}")
        sys.exit(1)
    
    # Print Results to Console (Logic ported from original main, but using report_data)
    if 'baseline' in report_data:
        print("\n[Naive (Baseline)]")
        print(f"  Latency: {report_data['baseline']['latency_sec']:.6f} sec")
        print(f"  Peak Mem: {report_data['baseline']['peak_memory_kb']:.2f} KB")
        
    if 'optimized' in report_data:
        print("\n[Optimized (Compiler)]")
        print(f"  Latency: {report_data['optimized']['latency_sec']:.6f} sec")
        print(f"  Peak Mem: {report_data['optimized']['peak_memory_kb']:.2f} KB")

    if 'comparison' in report_data:
        comp = report_data['comparison']
        print("\n" + "="*60)
        print("COMPARISON & ANALYSIS")
        print("="*60)
        print(f"Speedup: {comp['speedup_x']:.2f}x")
        print(f"Memory Efficiency: {comp['memory_savings_percent']:.1f}% reduction")
        print(f"Correctness Check: {'PASS' if comp['correctness'] else 'FAIL'} (Diff: {comp['max_diff']:.2e})")

    save_report(report_data)
    print("\nDone.")

if __name__ == "__main__":
    main()
