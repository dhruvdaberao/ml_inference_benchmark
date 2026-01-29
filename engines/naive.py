import numpy as np
from .base import ExecutionEngine
from typing import Dict, Any

class NaiveExecutionEngine(ExecutionEngine):
    """
    Standard NumPy implementation.
    Allocates new memory for every intermediate tensor.
    Easy to read, but memory inefficient.
    Simulates a 'framework' execution where the graph is dynamic or unoptimized.
    """
    
    def __init__(self, weights: Dict[str, np.ndarray], config: Dict[str, Any]):
        super().__init__(weights, config)
        self.name = "Naive (Baseline)"
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Execute forward pass using pure NumPy broadcasting and allocation.
        Each operation creates a new array.
        """
        # 1. Linear 1
        # Allocates new array for dot product results
        # Allocates another new array for the addition results
        hidden_pre = x @ self.W1 + self.b1
        
        # 2. ReLU
        # Allocates new array for the Result
        hidden = np.maximum(hidden_pre, 0)
        
        # 3. Linear 2
        # Allocates new array for dot product
        # Allocates new array for addition
        output = hidden @ self.W2 + self.b2
        
        return output

    def describe(self) -> str:
        return (
            "Naive (Baseline) Mode:\n"
            "  - Dynamic memory allocation for every operation.\n"
            "  - No pre-allocated buffers.\n"
            "  - Standard NumPy broadcasting (convenient but costly)."
        )
