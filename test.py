from black import re
import numpy as np

# Dense Layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(self.n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

    def reset(self):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(self.n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

    def adjust(self):
        # Adjust weights and biases
        self.weights += 0.05 * np.random.randn(self.n_inputs, self.n_neurons)
        self.biases += 0.05 * np.random.randn(1, self.n_neurons)

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputis, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create unitnitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flattten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class
class Loss:

    # Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards andy value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical lables
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Creaes activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate the gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:

    # Initialize optimizer
    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1 + self.decay * self.iterations)
            )

    # Update parameters
    def update_params(self, layer):

        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous updates multiplied by retain factor and update ith current gradients
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


def main():
    num_classes = 3
    iterations = 10_000
    X, y = spiral_data(samples=1000, classes=num_classes)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.savefig("test.png")

    # Init model
    dense1 = Layer_Dense(len(X[0]), 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, num_classes)
    # activation2 = Activation_Softmax()

    # Initialize loss function
    # loss_function = Loss_CategoricalCrossentropy()
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Initialize Stochastic Gradient Descent
    optimizer = Optimizer_SGD(learning_rate=1, decay=1e-3, momentum=0.99)

    tr = trange(iterations, desc="test", leave=True)
    for iteration in tr:

        # Forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        # Calculate loss
        loss = loss_activation.forward(dense2.output, y)

        # Calculate accuracy from output of model and targets
        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        tr.set_description(f"Iteration: {iteration}/{iterations}")
        tr.set_postfix(
            loss=loss,
            accuracy=accuracy,
            lr=optimizer.current_learning_rate,
            momentum=optimizer.momentum,
        )


if __name__ == "__main__":
    import nnfs
    from nnfs.datasets import spiral_data, vertical_data
    import matplotlib.pyplot as plt
    from tqdm import tqdm, trange

    nnfs.init()

    main()
