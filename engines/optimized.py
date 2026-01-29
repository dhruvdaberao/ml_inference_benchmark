import numpy as np
from .base import ExecutionEngine
from typing import Dict, Any

class OptimizedExecutionEngine(ExecutionEngine):
    """
    Compiler-optimized implementation simulation.
    Uses:
    - Pre-allocated output buffers (static memory planning)
    - In-place operations (out= argument)
    - Fused operations where possible
    
    Simulates how a compiled runtime (like XLA, TensorRT, or LiteRT) executes.
    """
    
    def __init__(self, weights: Dict[str, np.ndarray], config: Dict[str, Any]):
        super().__init__(weights, config)
        self.name = "Optimized (Compiler)"
        
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        
        max_batch_size = config.get('max_batch_size', 1)
        hidden_dim = config.get('hidden_dim', 4096)
        output_dim = config.get('output_dim', 1024)
        
        # MEMORY OPTIMIZATION: Static Buffer Allocation
        # These buffers are allocated ONCE and reused for every inference call.
        # This totally eliminates memory fragmentation and allocation overhead during run.
        self.hidden_buf = np.zeros((max_batch_size, hidden_dim), dtype=np.float32)
        self.output_buf = np.zeros((max_batch_size, output_dim), dtype=np.float32)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Execute forward pass using pre-allocated buffers and in-place ops.
        """
        batch_size = x.shape[0]
        
        # SAFETY: Ensure we don't exceed buffer size
        if batch_size > self.hidden_buf.shape[0]:
            raise ValueError(f"Batch size {batch_size} exceeds allocated buffer size {self.hidden_buf.shape[0]}")
            
        # Create views into the static buffers for the current batch size
        # Views are cheap (no allocation of data)
        current_hidden = self.hidden_buf[:batch_size]
        current_output = self.output_buf[:batch_size]
        
        # 1. Linear 1 (Fused MatMul)
        # Writes DIRECTLY into our pre-allocated 'current_hidden' buffer.
        # No temporary array created for the matmul result.
        np.dot(x, self.W1, out=current_hidden)
        
        # Bias Add (In-Place)
        # Adds b1 to current_hidden and stores result IN current_hidden.
        np.add(current_hidden, self.b1, out=current_hidden)
        
        # 2. ReLU (In-Place)
        # Computes relu and stores back in current_hidden.
        np.maximum(current_hidden, 0, out=current_hidden)
        
        # 3. Linear 2 (Fused MatMul)
        # Writes directly into the output buffer.
        np.dot(current_hidden, self.W2, out=current_output)
        
        # Bias Add (In-Place)
        np.add(current_output, self.b2, out=current_output)
        
        return current_output

    def describe(self) -> str:
        return (
            "Optimized (Compiler) Mode:\n"
            "  - Static buffer allocation (Zero allocs during run).\n"
            "  - In-place operations (np.dot(..., out=buf)).\n"
            "  - Simulates compiled graph execution."
        )
