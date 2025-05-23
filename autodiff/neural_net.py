import random
from typing import List
from autodiff.scalar import Scalar

class Module:
    def zero_grad(self) -> None:
        """
        Reset the gradients of all parameters to zero.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Scalar]:
        """
        Return a list of parameters of the module.
        """
        return []

class Neuron(Module):
    def __init__(self, num_inputs: int, use_relu=True):
        """
        Initialize the Neuron with the given number of inputs.

        :param num_inputs: Number of inputs that the neuron will receive
        :param use_relu: Whether to use ReLU activation function or no activation function
        """
        # We randomly initialize the weights of the neuron `self.w`
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(num_inputs)]
        # We initialize the bias `self.b` to 0
        self.b = Scalar(0)
        self.use_relu = use_relu

    def __call__(self, x: List[Scalar]) -> Scalar:
        """
        Forward pass through the neuron. Return a Scalar value, representing the output of the neuron.
        Apply the ReLU activation function if `self.use_relu` is True. Otherwise, use no activation function.
        Hint: Given a Scalar object `s`, you can compute the ReLU of `s` by calling `s.relu()`.

        :param x: List of Scalar values, representing the inputs to the neuron
        """
        # Compute the weighted sum: w1*x1 + w2*x2 + ... + wn*xn + b
        # Initialize accumulator with the bias
        out = self.b
        # Add each weighted input
        for i in range(len(self.w)):
            out = out + self.w[i] * x[i]
        
        # Apply ReLU activation if required
        if self.use_relu:
            out = out.relu()
        
        return out
        # TODO: Implement the forward pass through the neuron.
        raise NotImplementedError('Task 2.1 not implemented')

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.use_relu else 'Linear'}Neuron({len(self.w)})"

class FeedForwardLayer(Module):
    def __init__(self, num_inputs: int, num_outputs: int, use_relu: bool):
        """
        Initialize the FeedForwardLayer with the given number of inputs and outputs.

        :param num_inputs: Number of inputs that each neuron in that layer will receive
        :param num_outputs: Number of neurons in that layer
        """
        # TODO: Initialize the neurons in the layer. `self.neurons` should be a List of Neuron objects.
        # Create a list of neurons for this layer
        self.neurons = [Neuron(num_inputs, use_relu=use_relu) for _ in range(num_outputs)]
        #raise NotImplementedError('Task 2.2 not implemented')

    def __call__(self, x: List[Scalar]) -> List[Scalar]:
        """
        Forward pass through the layer. Return a list of Scalars, where each Scalar is the output of a neuron.

        :param x: List of Scalar values, representing the input features
        """
        #raise NotImplementedError('Task 2.2 not implemented')
        # Apply each neuron to the input
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"FeedForwardLayer of [{', '.join(str(n) for n in self.neurons)}]"

class MultiLayerPerceptron(Module):
    def __init__(self, num_inputs: int, num_hidden: List[int], num_outputs: int):
        """
        Initialize the MultiLayerPerceptron with the given architecture.
        Note that num_inputs and num_outputs are integers, while num_hidden is a list of integers.

        :param num_inputs: Number of input features
        :param num_hidden: List of integers, where each integer represents the number of neurons in that hidden layer
        :param num_outputs: Number of output neurons
        """
        # TODO: `self.layers` should be a List of FeedForwardLayer objects.
        #list created successfully
        self.layers = []
        # Input to 1st hidden layer
        if len(num_hidden) > 0:
            self.layers.append(FeedForwardLayer(num_inputs, num_hidden[0], use_relu=True))
            
            # Hidden layer to hidden layer
            for i in range(len(num_hidden) - 1):
                self.layers.append(FeedForwardLayer(num_hidden[i], num_hidden[i+1], use_relu=True))
            
            # Last hidden layer to output
            self.layers.append(FeedForwardLayer(num_hidden[-1], num_outputs, use_relu=False))
        else:
            # If no hidden layers, connect input directly to output
            self.layers.append(FeedForwardLayer(num_inputs, num_outputs, use_relu=False))        
        #raise NotImplementedError('Task 2.3 not implemented')

    def __call__(self, x: List[Scalar]) -> List[Scalar]:
        """
        Forward pass through the network.
        Call the first layer with the input x.
        Call each layer after that with the output of the previous layer.

        :param x: List of Scalar values, representing the input features
        """
        #Passing through layer forward
        for layer in self.layers:
            x = layer(x)
        return x
        #raise NotImplementedError('Task 2.3 not implemented')
        

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MultiLayerPerceptron of [{', '.join(str(layer) for layer in self.layers)}]"
