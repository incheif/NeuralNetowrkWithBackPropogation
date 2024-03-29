### MLP (Multi-Layer Perceptron) Implementation with Automatic Differentiation

This Python code implements a Multi-Layer Perceptron (MLP) using a custom `Value` class for automatic differentiation. The MLP is trained on a small dataset using stochastic gradient descent (SGD) with backpropagation.

#### Files:

- `mlp_autodiff.py`: This file contains the implementation of the MLP and related classes/functions.
  
#### Classes:

1. **`Value`**: 
   - Represents a scalar value with support for automatic differentiation.
   - Supports basic arithmetic operations (+, -, *, /) and some transcendental functions (tanh, exp).
   
2. **`Neuron`**: 
   - Represents a single neuron in a neural network layer.
   - Initialized with random weights and bias.
   - Utilizes the `Value` class to compute the output of the neuron and its derivative during backpropagation.

3. **`Layer`**: 
   - Represents a layer of neurons in the neural network.
   - Composed of multiple neurons.
   - Computes the output of each neuron in the layer.

4. **`MLP`**: 
   - Represents the Multi-Layer Perceptron model.
   - Composed of multiple layers.
   - Computes the output of the entire network given an input.
   
#### Usage:

1. **Training Data**:
   - Training data consists of input-output pairs (`xs`, `ys`).
   - Each input (`xs`) is a list of features.
   - Each output (`ys`) is a single scalar value representing the target.

2. **Training Loop**:
   - The code contains a training loop that iterates over the dataset for a fixed number of epochs.
   - In each iteration, it computes the predicted output (`ypred`) of the MLP for each input and calculates the loss using a squared error loss function.
   - It then computes gradients using backpropagation and updates the model parameters using SGD.

3. **Initialization**:
   - The MLP is initialized with the number of input features and the number of neurons in each layer.

#### Dependencies:

- `math`: Standard Python math library for mathematical functions.
- `random`: Standard Python library for generating random numbers.
- `numpy`: Required for array operations.
- `matplotlib.pyplot`: Required for plotting graphs.

#### Running the Code:

- The MLP can be instantiated with the desired architecture by providing the number of input features and the number of neurons in each layer.
- The training data (`xs`, `ys`) can be modified or expanded to fit the specific problem.

#### Further Improvements:

- Currently, the code supports only regression tasks with a single output.
- Additional activation functions (e.g., ReLU, sigmoid) and loss functions (e.g., cross-entropy) could be implemented for classification tasks.
- Hyperparameters such as learning rate, number of epochs, and network architecture can be tuned for better performance.
- The code could be extended to support different optimization algorithms apart from SGD.
