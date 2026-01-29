import numpy as np

class MLPModel:
    """
    A simple Multi-Layer Perceptron (MLP) specifically designed for
    inference benchmarking.
    
    Architecture:
        Input (batch_size, input_dim)
          -> Linear(input_dim -> hidden_dim)
          -> ReLU
          -> Linear(hidden_dim -> output_dim)
          -> Output
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        """
        Initialize the model weights deterministically.
        
        Args:
            input_dim: size of input features
            hidden_dim: size of hidden layer
            output_dim: size of output features
            seed: random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        
        # Initialize weights and biases
        # W1 shape: (input_dim, hidden_dim)
        # b1 shape: (hidden_dim,)
        self.W1 = self.rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # W2 shape: (hidden_dim, output_dim)
        # b2 shape: (output_dim,)
        self.W2 = self.rng.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        
    def get_weights(self):
        """Return a dictionary of weights for execution runners."""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
