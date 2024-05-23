# Import NumPy for numerical operations and array manipulations
import numpy as np

# Import the PyTorch library
import torch

# Import the neural network module from PyTorch
import torch.nn as nn

# +++++++++++++++++++++++++++++ FOURIER FEATURES +++++++++++++++++++++++++++++


class NeuralFieldFF(nn.Module):
    """
    Define the NeuralFieldMLP class, which inherits from nn.Module
    """

    # Initialize the class with default parameters for input size,
    # number of hidden layers, and layer size
    def __init__(self, input_size=256, hidden_layers=2, layer_size=256):
        # Call the parent class's constructor
        super().__init__()

        # Create an empty list to hold the layers
        layers = [
            # Add the input layer
            nn.Linear(input_size, layer_size),
            # Uncomment for batch normalization
            # nn.BatchNorm1d(layer_size),
            # Add ReLU activation function
            nn.ReLU()
        ]

        # Add hidden layers
        for _ in range(hidden_layers):
            layers.extend([
                # Add a hidden layer
                nn.Linear(layer_size, layer_size),
                # Uncomment for batch normalization
                # nn.BatchNorm1d(layer_size),
                # Add ReLU activation function
                nn.ReLU()
            ])

        # Add the output layer
        layers.append(nn.Linear(layer_size, 1))
        # Add the Sigmoid activation function
        layers.append(nn.Sigmoid())

        # Create the neural network model as a sequence of layers
        self.model = nn.Sequential(*layers)

    # Define the forward pass
    def forward(self, input):
        # Check the dimension of the input and permute if necessary
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1)

        # Pass the input through the neural network
        input = self.model(input)

        # Check the dimension of the output and permute back if necessary
        if input.dim() == 4:
            input = input.permute(0, 3, 1, 2)

        # Return the output
        return input

# +++++++++++++++++++++++++++++ SIREN +++++++++++++++++++++++++++++


class SineLayer(nn.Module):
    """
    This class defines a Sine Layer, which is a key component of the SIREN architecture.
    The parameter omega_0 controls the frequency of the sine activation function.
    If is_first is True, omega_0 acts as a frequency factor for the first layer.
    If is_first is False, the weights are scaled by omega_0 to maintain activation magnitudes.
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    # Initialize the weights of the layer
    def init_weights(self):
        with torch.no_grad():
            # If this is the first layer, initialize weights uniformly between -1/in_features and 1/in_features
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            # Otherwise, initialize weights to maintain activation magnitudes
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    # Forward pass through the layer
    def forward(self, input):
        # Apply the sine activation function, scaled by omega_0
        return torch.sin(self.omega_0 * self.linear(input))


class NeuralFieldSiren(nn.Module):
    """
    This class defines the SIREN model, which is a neural network composed of Sine Layers.
    The model takes several parameters including the number of input features, hidden features,
    hidden layers, and output features. It also allows for the outermost layer to be linear.
    The frequency omega_0 can be set differently for the first and hidden layers.
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        layers = []
        # Add the first Sine Layer
        layers.append(SineLayer(in_features, hidden_features,
                                is_first=True, omega_0=first_omega_0))

        # Add hidden Sine Layers
        for i in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))

        # Add the final layer, which can be either linear or another Sine Layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features,
                                    is_first=False, omega_0=hidden_omega_0))

        # Convert the list of layers into a PyTorch Sequential model
        self.model = nn.Sequential(*layers)

    # Forward pass through the network
    def forward(self, input):
        # Clone and detach the coordinates to enable gradient computation w.r.t. input
        input = input.clone().detach().requires_grad_(True)

        # Check the dimension of the input and permute if necessary
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1)

        # Pass the input through the neural network
        output = self.model(input)

        # Check the dimension of the output and permute back if necessary
        if output.dim() == 4:
            output = output.permute(0, 3, 1, 2)

        # Return the output
        return output