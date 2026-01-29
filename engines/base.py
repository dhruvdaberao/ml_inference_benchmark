from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class ExecutionEngine(ABC):
    """
    Abstract base class for all execution engines.
    Enforces a common interface for running inference and retrieving stats.
    """
    
    def __init__(self, weights: Dict[str, np.ndarray], config: Dict[str, Any]):
        """
        Initialize the engine with model weights and configuration.
        
        Args:
            weights: Dictionary of model weights (W1, b1, w2, b2, etc.)
            config: Dictionary containing 'max_batch_size', 'input_dim', etc.
        """
        self.weights = weights
        self.config = config
        self.name = "AbstractEngine"
        
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Execute the forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        pass
        
    def describe(self) -> str:
        """Return a human-readable description of how this engine executes."""
        return f"{self.name}: Base execution engine."
