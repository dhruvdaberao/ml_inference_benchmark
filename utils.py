import json
import os
import numpy as np
from typing import Dict, Any, List

def parse_input_string(input_str: str) -> List[float]:
    """Parse comma-separated string into list of floats."""
    try:
        return [float(x.strip()) for x in input_str.split(',')]
    except ValueError:
        raise ValueError("Invalid input format. Expected comma-separated floats (e.g., '1.0, 0.5, -0.2').")

def tile_input_data(user_input: List[float], batch_size: int, input_dim: int) -> np.ndarray:
    """
    Create a full input tensor from user input.
    - If user input is shorter than input_dim, loop it.
    - If longer, truncate.
    - Replicate across batch dimension.
    """
    if not user_input:
        raise ValueError("Input data cannot be empty.")
        
    # 1. Create a single feature vector of length input_dim
    feature_vector = np.array(user_input, dtype=np.float32)
    
    # Tile or truncate to match input_dim
    if len(feature_vector) < input_dim:
        repeats = (input_dim // len(feature_vector)) + 1
        feature_vector = np.tile(feature_vector, repeats)[:input_dim]
    else:
        feature_vector = feature_vector[:input_dim]
        
    # 2. Replicate across batch
    # Shape: (batch_size, input_dim)
    batch_data = np.tile(feature_vector, (batch_size, 1))
    
    return batch_data

def format_summary(values: np.ndarray) -> str:
    """Return a pretty string summary of the output."""
    return (
        f"  Min:  {values.min():.4f}\n"
        f"  Max:  {values.max():.4f}\n"
        f"  Mean: {values.mean():.4f}\n"
        f"  First 5 values: {values.flatten()[:5].tolist()}"
    )

def save_report(report_data: Dict[str, Any], output_dir: str = "reports"):
    """Save analysis results to JSON and TXT."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(output_dir, "analysis_report.json")
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
        
    # TXT Human Readable
    txt_path = os.path.join(output_dir, "analysis_report.txt")
    with open(txt_path, 'w') as f:
        f.write("ML EXECUTION ANALYSIS REPORT\n")
        f.write("============================\n\n")
        
        # Write generic info
        f.write(f"Timestamp: {report_data.get('timestamp')}\n")
        f.write(f"Batch Size: {report_data.get('config', {}).get('batch_size')}\n")
        f.write(f"Mode: {report_data.get('mode')}\n\n")
        
        # Write results if present
        if 'baseline' in report_data:
            f.write("BASELINE EXECUTION\n")
            f.write("------------------\n")
            f.write(f"Latency: {report_data['baseline']['latency_sec']:.6f} sec\n")
            f.write(f"Peak Mem: {report_data['baseline']['peak_memory_kb']:.2f} KB\n\n")
            
        if 'optimized' in report_data:
            f.write("OPTIMIZED EXECUTION\n")
            f.write("-------------------\n")
            f.write(f"Latency: {report_data['optimized']['latency_sec']:.6f} sec\n")
            f.write(f"Peak Mem: {report_data['optimized']['peak_memory_kb']:.2f} KB\n\n")
            
        if 'comparison' in report_data:
            f.write("COMPARISON SUMMARY\n")
            f.write("------------------\n")
            c = report_data['comparison']
            f.write(f"Speedup: {c.get('speedup_x', 0):.2f}x\n")
            f.write(f"Memory Saved: {c.get('memory_savings_percent', 0):.1f}%\n")
            f.write(f"Correctness: {'PASS' if c.get('correctness') else 'FAIL'}\n")

    print(f"\n[Artifacts] Reports saved to: {output_dir}/")
