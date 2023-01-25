from dataclasses import dataclass
import torch


class SimpleMLP(torch.nn.Sequential):
    """Simple MLP model for image classification."""

    def __init__(self, num_classes: int, hidden_dim: int, input_dim: int = 784):
        """Constructor for a simple, ReLU activated, 1 hidden layer MLP.

        Args:
          num_classes: Number of class to predict.
          hidden_dim: Size of the hidden layer.
          input_dim: Size of the flatten image.
        """
        super().__init__(
            torch.nn.Flatten(),
            torch.nn.Linear(
                input_dim, hidden_dim,
            ),  # The input size for the linear layer is determined by the previous operations
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden_dim, num_classes
            ),  # Here we get exactly num_classes logits at the output
        )


@dataclass
class SimpleMLPFactory:
    """Factory for SimpleMLP."""
    num_classes: int
    "Number of classes."

    hidden_dim: int
    "Size of the hidden layer."

    def __call__(self) -> SimpleMLP:
        """Build a SimpleMLP with the parameters."""
        return SimpleMLP(self.num_classes, self.hidden_dim)
