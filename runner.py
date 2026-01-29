import numpy as np

class BaselineRunner:
    """
    Standard NumPy implementation.
    Allocates new memory for every intermediate tensor.
    Easy to read, but memory inefficient.
    """
    def __init__(self, weights: dict):
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 1. Linear 1
        # Allocates new array for dot product, then another for addition
        hidden_pre = x @ self.W1 + self.b1
        
        # 2. ReLU
        # Allocates new array for maximum
        hidden = np.maximum(hidden_pre, 0)
        
        # 3. Linear 2
        # Allocates new array for dot product, then another for addition
        output = hidden @ self.W2 + self.b2
        
        return output

class OptimizedRunner:
    """
    Compiler-optimized implementation simulation.
    Uses:
    - Pre-allocated output buffers (static memory planning)
    - In-place operations (out= argument)
    - Fused operations where possible (np.add with out)
    """
    def __init__(self, weights: dict, max_batch_size: int, input_dim: int, hidden_dim: int, output_dim: int):
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        
        # KEY OPTIMIZATION: Static Buffer Allocation
        # allocate buffers once during init, reuse them every inference
        self.hidden_buf = np.zeros((max_batch_size, hidden_dim), dtype=np.float32)
        self.output_buf = np.zeros((max_batch_size, output_dim), dtype=np.float32)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        
        # Use views of buffers to handle variable batch size (up to max_alloc)
        # In a real compiler, we might demand fixed batch size or re-alloc if exceeded.
        current_hidden = self.hidden_buf[:batch_size]
        current_output = self.output_buf[:batch_size]
        
        # 1. Linear 1 (Fused MatMul + Bias Add)
        # x @ W1 -> writes directly into current_hidden
        np.dot(x, self.W1, out=current_hidden)
        
        # Bias Add: In-place add b1 to current_hidden
        # np.add(a, b, out=a) is a common compiler pattern
        np.add(current_hidden, self.b1, out=current_hidden)
        
        # 2. ReLU (In-place)
        # writes result back into current_hidden, saving an allocation
        np.maximum(current_hidden, 0, out=current_hidden)
        
        # 3. Linear 2 (Fused MatMul + Bias Add)
        # hidden @ W2 -> writes directly into current_output
        np.dot(current_hidden, self.W2, out=current_output)
        
        # Bias Add: In-place add b2 to current_output
        np.add(current_output, self.b2, out=current_output)
        
        return current_output
