# ML Execution & Optimization Analyzer

**A professional tool for analyzing inference execution behavior and the impact of compiler-style optimizations.**

This tool simulates how machine learning compilers (like XLA, TensorRT) optimize graph execution compared to standard eager execution frameworks (like PyTorch/NumPy). It provides detailed profiling of memory usage, latency, and buffer allocation strategies.

## 🚀 Key Features

- **Real-World Simulation**: Compare "Naive" (Eager) vs. "Optimized" (Static Buffer) execution modes.
- **Detailed Profiling**: Track peak memory usage, execution latency, and buffer allocations.
- **CLI Interface**: Easy-to-use command line interface for custom inputs and batch sizes.
- **Correctness Verification**: Automatically verifies that optimizations do not degrade numerical precision.
- **Structured Reporting**: Generates JSON and TXT reports for analysis.

## 🛠 Installation

1. Clone the repository.
2. Install dependencies (standard Python 3.8+):
   ```bash
   pip install numpy
   ```

## 💻 Usage

Run the analyzer using the `analyze.py` script.

### Basic Execution
```bash
python analyze.py --input "0.5, 1.2, -0.3" --batch 32
```

### Options
- `--input`: Comma-separated list of float values (e.g., `"1.0, 0.5"`). The tool automatically tiles this input to match the model's expected dimension (1024).
- `--batch`: Batch size (default: 32). Increase this to see larger memory savings.
- `--mode`: `baseline` | `optimized` | `both` (default: `both`).

### Example Output

```text
ML EXECUTION ANALYZER
============================================================
Mode: BOTH
Batch Size: 32
Model: MLP (1024 -> 4096 -> 1024)
...
COMPARISON & ANALYSIS
============================================================
Speedup: 1.15x
Memory Efficiency: 97.5% reduction
Correctness Check: PASS (Diff: 0.00e+00)
```

## 🧠 Theory: Why Optimization Matters

### Naive Execution (Baseline)
In standard frameworks (like NumPy or PyTorch Eager), every operation (e.g., `A + B`) allocates a **new memory buffer** for the result.
- **Pros**: Flexible, easy to debug.
- **Cons**: High memory fragmentation, frequent allocations, poor cache locality.

### Optimized Execution (Compiler)
Compilers perform **Static Memory Planning**. They analyze the graph ahead of time and pre-allocate a fixed set of buffers.
- **Techniques**:
    - **Buffer Reuse**: Reusing the same memory for different tensors when their lifetimes don't overlap.
    - **In-Place Operations**: Writing results directly into pre-allocated outputs (`out=buffer`).
    - **Fusion**: Combining multiple operations to avoid strictly reading/writing to memory.
- **Result**: drastically reduced peak memory usage and improved latency.

## 📂 Project Structure

- `engines/`: Core execution logic.
    - `naive.py`: Eager execution simulation.
    - `optimized.py`: Compiler-style execution simulation.
- `analyze.py`: CLI entry point.
- `profiler.py`: Memory and latency measurement tools.
- `reports/`: Generated analysis artifacts.

---
*Built for the Advanced Agentic Coding Interview.*
