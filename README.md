# ML Inference Benchmarking & Analysis Framework

A standalone, pure Python project designed to demonstrate the impact of **compiler-style optimizations** on neural network inference performance. 

This project simulates how an AI compiler (like XLA, TVM, or TensorRT) transforms a computational graph to run faster and use less memory, without relying on any external ML frameworks like PyTorch or TensorFlow.

## 🎯 Project Goals
- **Measure execution latency**: Compare naive NumPy code vs. optimized kernels.
- **Measure memory pressure**: Quantify the benefits of static memory planning.
- **Explain optimizations**: Demonstrate **Operator Fusion**, **In-place Operations**, and **Static Buffer Allocation**.

## 🏗️ Architecture
The framework executes a simple MLP (Linear -> ReLU -> Linear) in two modes:

### 1. Baseline Mode (`BaselineRunner`)
- Simulates a standard eager execution framework (like PyTorch Eager or standard NumPy).
- **Behavior**: Allocates new memory for every intermediate operation.
- **Pros**: Simple to write and debug.
- **Cons**: High memory framgmentation and allocation overhead.

### 2. Optimized Mode (`OptimizedRunner`)
- Simulates a compiled runtime.
- **Static Buffer Allocation**: All intermediate tensors (`hidden_buf`, `output_buf`) and outputs are pre-allocated at startup. No `malloc` during inference.
- **In-Place Operations**: Uses `out=` arguments in NumPy to write directly into buffers, avoiding temporaries.
- **Operator Fusion**: Fuses Linear and Arithmetic operations where possible to keep data in cache.

## 🚀 How to Run

1. **Environment Setup** (Python 3.8+)
   No external ML libraries required. Just NumPy.
   ```bash
   pip install numpy
   ```

2. **Run the Benchmark**
   ```bash
   python ml_inference_benchmark/benchmark.py
   ```

## 📊 Sample Output
*(See `results.txt` for a full run log)*

```text
Baseline:
  Time: 0.015200 sec
  Peak Memory: 4096.00 KB

Optimized:
  Time: 0.008900 sec
  Peak Memory: 1024.00 KB

⚡ Performance Win:
  Speedup: 1.71x faster
  Memory:  75.0% reduction in peak allocation
```

## 🧠 Engineering Insights
- **Memory Latency**: A significant portion of "compute" time in bandwidth-bound workloads (like MLPs on CPU) is actually just memory allocation and data movement.
- **Predictability**: The optimized runner has zero allocations during the forward pass, making latency highly predictable—critical for real-time systems.

---
*Created for an AI Compiler Engineering Systems role.*
