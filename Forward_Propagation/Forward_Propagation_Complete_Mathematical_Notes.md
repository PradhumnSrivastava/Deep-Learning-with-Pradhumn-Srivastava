# Forward Propagation: Complete Mathematical and Conceptual Notes

## From Beginner to Advanced \| Deep Learning \| Mathematical Derivations \| NumPy Implementation

------------------------------------------------------------------------

## Table of Contents

1.  Introduction to Forward Propagation
2.  Why Do We Need Forward Propagation?
3.  Neural Network Refresher
4.  The Complete Journey of Data Through a Network
5.  Mathematical Notation and Symbols
6.  Forward Propagation in a Single Neuron
7.  Weighted Sum, Bias, and Pre-Activation
8.  Activation Functions Used During Forward Propagation
9.  Forward Propagation in a One-Hidden-Layer Network
10. Matrix Form of Forward Propagation
11. Understanding Matrix and Vector Dimensions
12. Complete Numerical Example: One Hidden Layer
13. Forward Propagation for Multiple Examples (Batch Processing)
14. Forward Propagation for Binary Classification
15. Forward Propagation for Multiclass Classification and Softmax
16. Forward Propagation for Regression
17. Forward Propagation in Deep Neural Networks
18. Forward Propagation in Tensor Form
19. NumPy Implementation From Scratch
20. Forward Propagation Using Keras
21. Forward Propagation During Training vs. Inference
22. Common Mistakes and Debugging Checklist
23. Computational Cost and Numerical Stability
24. Common Interview Questions
25. Practice Problems
26. Summary and Key Formulas

------------------------------------------------------------------------

# 1. Introduction to Forward Propagation

## 1.1 What Is Forward Propagation?

**Forward propagation is the process of passing input data through a
neural network, layer by layer, to calculate the network's prediction.**

The data moves in the forward direction:

$$
\text{Input} \rightarrow \text{Hidden Layer(s)} \rightarrow \text{Output Layer} \rightarrow \text{Prediction}
$$

At each layer, the network generally performs two main operations:

1.  Calculate a weighted sum and add a bias.
2.  Apply an activation function.

For neuron (j) in layer (l), the calculation is:

$$
z_j^{[l]} = \sum_{i=1}^{n_{l-1}} w_{ji}^{[l]}a_i^{[l-1]} + b_j^{[l]}
$$

Then:

$$
a_j^{[l]} = g^{[l]}(z_j^{[l]})
$$

Here, (z) is the pre-activation value and (a) is the activation (the
output produced by the neuron).

The output of one layer becomes the input to the next layer.

## 1.2 A Simple Analogy

Imagine a student evaluation system:

-   Inputs: attendance, assignment score, and exam score.
-   Hidden layer: combines these signals into learned patterns.
-   Output layer: estimates the probability that the student passes.

Each neuron combines incoming information using weights and a bias. The
activation function transforms that combined value. The final layer
produces the prediction.

This is only an analogy; real neural networks learn weights from data
rather than following manually written rules.

## 1.3 Is Forward Propagation the Same as Learning?

No. Forward propagation computes a prediction using the current
parameters. It does not, by itself, update the weights.

A typical training iteration contains:

$$
\text{Forward Pass} \rightarrow \text{Loss Calculation} \rightarrow
\text{Backpropagation} \rightarrow \text{Parameter Update}
$$

-   **Forward propagation:** What prediction does the network make?
-   **Loss function:** How far is that prediction from the target?
-   **Backpropagation:** How should the loss be differentiated with
    respect to parameters?
-   **Optimizer:** How should parameters be updated using those
    gradients?

Forward propagation is necessary for training, but it is also used when
a trained model makes predictions.

------------------------------------------------------------------------

# 2. Why Do We Need Forward Propagation?

A neural network contains parameters---weights and biases---that define
how input features are transformed. Forward propagation uses those
parameters to map an input to an output.

Examples:

  -----------------------------------------------------------------------
  Task                    Input                   Network output
  ----------------------- ----------------------- -----------------------
  House-price prediction  Area, rooms, location   Predicted price
                          features                

  Spam classification     Text-derived features   Spam probability

  Image classification    Pixel values            Class scores or
                                                  probabilities

  Sentiment analysis      Token representations   Sentiment scores

  Medical-risk estimation Patient features        Estimated risk
  -----------------------------------------------------------------------

A network can have many layers, but the principle remains the same: each
layer transforms the representation received from the previous layer.

## 2.1 Forward Propagation as a Function

A neural network represents a parameterized function:

$$
\hat{y} = f_{\theta}(x)
$$

where:

-   \(x\) is the input.
-   (`\hat{y}`{=tex}) is the prediction.
-   (`\theta`{=tex}) represents all learnable weights and biases.
-   (f\_{`\theta`{=tex}}) is the complete network computation.

For a network with (L) layers:

$$
a^{[0]} = x
$$

$$
a^{[l]} = g^{[l]}\left(W^{[l]}a^{[l-1]} + b^{[l]}\right),
\quad l=1,2,\ldots,L
$$

The final prediction is:

$$
\hat{y} = a^{[L]}
$$

This compact expression is the central formula for forward propagation.

------------------------------------------------------------------------

# 3. Neural Network Refresher

## 3.1 What Is a Neural Network?

An artificial neural network is a collection of interconnected
computational units called neurons. Neurons are organized into layers.

A common fully connected network contains:

1.  **Input layer:** Receives the features.
2.  **Hidden layer(s):** Learn intermediate representations.
3.  **Output layer:** Produces the final prediction.

Example architecture:

$$
3 \rightarrow 4 \rightarrow 2
$$

This means: - 3 input features. - 4 neurons in one hidden layer. - 2
output neurons.

The numbers describe neuron counts, not the values of the input data.

## 3.2 Weights

Weights control how strongly an input contributes to a neuron's
calculation.

For a neuron with two inputs:

$$
z = w_1x_1 + w_2x_2 + b
$$

If (w_1) is large and positive, increasing (x_1) tends to increase (z),
assuming other values remain fixed. A negative weight tends to move (z)
in the opposite direction.

A weight does not necessarily express causal importance. Its
interpretation depends on the other parameters, feature scaling, and
network structure.

## 3.3 Bias

The bias is an additive parameter:

$$
z = wx+b
$$

Without a bias, the pre-activation is forced to be zero when all inputs
are zero. A bias gives the neuron additional flexibility by shifting its
response.

## 3.4 Activation

The activation function transforms the pre-activation:

$$
a = g(z)
$$

For example, ReLU is:

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

If (z=-2), the activation is (0). If (z=3), the activation is (3).

## 3.5 Parameters vs. Activations

These terms should not be confused:

-   **Parameters:** weights and biases, usually learned during training.
-   **Activations:** intermediate values computed for a particular
    input.
-   **Hyperparameters:** choices such as number of layers, learning
    rate, batch size, and activation type.

Forward propagation uses parameters to compute activations.

------------------------------------------------------------------------

# 4. The Complete Journey of Data Through a Network

Consider a network with input (x), a hidden layer, and an output layer.

## Step 1: Receive the input

For example:

$$
x =
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$

The input has two features.

## Step 2: Calculate the hidden layer's weighted sums

For two hidden neurons:

$$
z_1^{[1]}=w_{11}^{[1]}x_1+w_{12}^{[1]}x_2+b_1^{[1]}
$$

$$
z_2^{[1]}=w_{21}^{[1]}x_1+w_{22}^{[1]}x_2+b_2^{[1]}
$$

## Step 3: Apply the hidden activation function

If the hidden layer uses ReLU:

$$
a_1^{[1]}=\max(0,z_1^{[1]})
$$

$$
a_2^{[1]}=\max(0,z_2^{[1]})
$$

These activations become the input to the next layer.

## Step 4: Calculate the output layer

For one output neuron:

$$
z^{[2]}=w_1^{[2]}a_1^{[1]}+w_2^{[2]}a_2^{[1]}+b^{[2]}
$$

If this is a binary-classification output, sigmoid may be used:

$$
\hat{y}=\sigma(z^{[2]})
$$

## Step 5: Interpret the output

The sigmoid output lies between 0 and 1 and can be interpreted as a
model-estimated probability for the positive class, assuming the model
is used and calibrated appropriately.

A threshold such as (0.5) can convert the probability into a class
label, but the threshold is a decision rule, not part of the sigmoid
calculation itself.

------------------------------------------------------------------------

# 5. Mathematical Notation and Symbols

  Symbol              Meaning
  ------------------- ---------------------------------------
  \(x\)               Input feature vector
  \(y\)               Actual target
  (`\hat{y}`{=tex})   Model prediction
  (W\^{\[l\]})        Weight matrix for layer (l)
  (b\^{\[l\]})        Bias vector for layer (l)
  (z\^{\[l\]})        Pre-activation vector for layer (l)
  (a\^{\[l\]})        Activation vector for layer (l)
  (g\^{\[l\]})        Activation function used at layer (l)
  (n_l)               Number of neurons in layer (l)
  \(L\)               Number of trainable layers
  \(m\)               Number of examples in a batch
  (`\theta`{=tex})    All network parameters collectively

## 5.1 Layer Index Notation

The notation (W\^{\[2\]}) means the weight matrix for layer 2. The
square-bracket index identifies the layer; it is not an exponent.

Similarly, (a\^{\[1\]}) means the activation vector from layer 1.

The convention used here is:

$$
a^{[0]}=x
$$

Layer 0 is the input representation. The first trainable layer is layer
1.

## 5.2 One Neuron vs. a Whole Layer

For one neuron:

$$
z_j^{[l]} = \sum_{i=1}^{n_{l-1}} w_{ji}^{[l]}a_i^{[l-1]} + b_j^{[l]}
$$

For the entire layer:

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

Then:

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

For elementwise activations such as ReLU and sigmoid, the function is
applied to each element of the vector.

------------------------------------------------------------------------

# 6. Forward Propagation in a Single Neuron

A single neuron is the smallest useful example.

Suppose:

$$
x_1=2,\qquad x_2=3
$$

$$
w_1=0.5,\qquad w_2=-0.2,\qquad b=0.1
$$

The pre-activation is:

$$
z=w_1x_1+w_2x_2+b
$$

Substitute the values:

$$
z=(0.5)(2)+(-0.2)(3)+0.1
$$

$$
z=1-0.6+0.1=0.5
$$

If the activation is ReLU:

$$
a=\max(0,0.5)=0.5
$$

If the activation is sigmoid instead:

$$
a=\sigma(0.5)=\frac{1}{1+e^{-0.5}}\approx 0.6225
$$

The weighted sum and bias are the same in both cases. Only the
activation function changes the final output.

## 6.1 Vector Form

Write the input and weights as vectors:

$$
x=
\begin{bmatrix}
2\\
3
\end{bmatrix},
\qquad
w=
\begin{bmatrix}
0.5\\
-0.2
\end{bmatrix}
$$

Then:

$$
z=w^Tx+b
$$

Here:

$$
w^Tx=
\begin{bmatrix}
0.5 & -0.2
\end{bmatrix}
\begin{bmatrix}
2\\
3
\end{bmatrix}
=0.5(2)-0.2(3)=0.4
$$

Adding the bias gives:

$$
z=0.4+0.1=0.5
$$

The dot product (w\^Tx) is a compact way to write the weighted sum.

------------------------------------------------------------------------

# 7. Weighted Sum, Bias, and Pre-Activation

## 7.1 Weighted Sum

For (n) inputs:

$$
s=\sum_{i=1}^{n}w_ix_i
$$

Every input (x_i) is multiplied by its corresponding weight (w_i), and
the products are added.

## 7.2 Add the Bias

$$
z=s+b=\sum_{i=1}^{n}w_ix_i+b
$$

The value (z) is often called the **pre-activation** or **linear
output**.

## 7.3 Apply the Activation

$$
a=g(z)
$$

The result (a) is called the activation or post-activation.

The full neuron computation is therefore:

$$
\boxed{a=g\left(\sum_{i=1}^{n}w_ix_i+b\right)}
$$

## 7.4 Why Separate (z) and (a)?

Separating these values helps with reasoning and debugging:

-   \(z\) tells us the result before the nonlinearity.
-   \(a\) tells us the value passed to the next layer.

For ReLU, a negative (z) becomes a zero activation. For sigmoid, any
finite real (z) becomes a value strictly between 0 and 1.

------------------------------------------------------------------------

# 8. Activation Functions Used During Forward Propagation

## 8.1 Step Function

The step function is commonly associated with the classic perceptron:

$$
g(z)=
\begin{cases}
1, & z\geq 0\\
0, & z<0
\end{cases}
$$

It creates a hard threshold. It is useful for understanding a
perceptron, but its zero derivative almost everywhere makes it
unsuitable for ordinary gradient-based training of hidden layers.

## 8.2 Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Properties:

-   Output lies in ((0,1)) for finite (z).
-   Often used for a single binary-classification output.
-   Can saturate near 0 or 1, producing small gradients.

Example:

$$
\sigma(0)=0.5
$$

$$
\sigma(2)\approx 0.8808
$$

$$
\sigma(-2)\approx 0.1192
$$

Its derivative is:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

## 8.3 Hyperbolic Tangent (tanh)

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

Its output lies in ((-1,1)). It is zero-centered, but it can also
saturate for large positive or negative inputs.

## 8.4 ReLU

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

Piecewise form:

$$
\operatorname{ReLU}(z)=
\begin{cases}
z, & z>0\\
0, & z\leq 0
\end{cases}
$$

ReLU is widely used in hidden layers because it is simple and does not
saturate on the positive side. A neuron that repeatedly receives
negative pre-activations can, however, become inactive; this is often
called the "dying ReLU" problem.

## 8.5 Leaky ReLU

$$
\operatorname{LeakyReLU}(z)=
\begin{cases}
z, & z\geq 0\\
\alpha z, & z<0
\end{cases}
$$

Here (`\alpha`{=tex}) is a small positive slope, such as (0.01), chosen
as a hyperparameter.

## 8.6 Softmax

For a vector of class logits (z), softmax converts the logits into
normalized positive values:

$$
\operatorname{softmax}(z)_i=
\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

The outputs sum to 1:

$$
\sum_{i=1}^{K}\operatorname{softmax}(z)_i=1
$$

Softmax is commonly used for mutually exclusive multiclass
classification. In practice, implementations generally use a numerically
stable version that subtracts the maximum logit before exponentiation.

## 8.7 Choosing an Output Activation

  -----------------------------------------------------------------------
  Problem                 Common output           Typical output
                          activation              interpretation
  ----------------------- ----------------------- -----------------------
  Regression              Linear / no activation  Real-valued prediction

  Binary classification   Sigmoid                 Positive-class
                                                  probability

  Multiclass, one class   Softmax                 Distribution over
  per example                                     classes

  Multilabel              Independent sigmoid     One probability per
  classification          outputs                 label
  -----------------------------------------------------------------------

The appropriate choice depends on the target and loss function. The
activation and loss must be compatible.

------------------------------------------------------------------------

# 9. Forward Propagation in a One-Hidden-Layer Network

Consider a network with: - 2 input features. - 2 hidden neurons. - 1
output neuron.

The architecture is:

$$
2 \rightarrow 2 \rightarrow 1
$$

Let the input be:

$$
x=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$

## 9.1 Hidden Layer Calculation

For hidden neuron 1:

$$
z_1^{[1]}=w_{11}^{[1]}x_1+w_{12}^{[1]}x_2+b_1^{[1]}
$$

For hidden neuron 2:

$$
z_2^{[1]}=w_{21}^{[1]}x_1+w_{22}^{[1]}x_2+b_2^{[1]}
$$

Using ReLU:

$$
a_1^{[1]}=\operatorname{ReLU}(z_1^{[1]})
$$

$$
a_2^{[1]}=\operatorname{ReLU}(z_2^{[1]})
$$

## 9.2 Output Layer Calculation

The output neuron receives the two hidden activations:

$$
z^{[2]}=w_{11}^{[2]}a_1^{[1]}+w_{12}^{[2]}a_2^{[1]}+b^{[2]}
$$

For binary classification:

$$
\hat{y}=\sigma(z^{[2]})
$$

Notice that the output layer receives the **hidden activations**, not
the original input features directly (in this simple architecture).

## 9.3 Composition of Functions

The network can be written as a nested function:

$$
\hat{y}=
\sigma\left(
W^{[2]}\operatorname{ReLU}(W^{[1]}x+b^{[1]})+b^{[2]}
\right)
$$

This expression shows that the output is a composition of
transformations. The hidden layer transforms the original input into a
learned representation, and the output layer transforms that
representation into a prediction.

------------------------------------------------------------------------

# 10. Matrix Form of Forward Propagation

Matrix notation makes calculations concise and is the form commonly used
in deep-learning libraries.

## 10.1 Input Vector

For (n_0) input features:

$$
a^{[0]}=x\in\mathbb{R}^{n_0}
$$

## 10.2 Weight Matrix

For layer (l), with (n\_{l-1}) inputs and (n_l) neurons:

$$
W^{[l]}\in\mathbb{R}^{n_l\times n_{l-1}}
$$

Each row of (W\^{\[l\]}) contains the weights feeding into one neuron of
layer (l).

## 10.3 Bias Vector

$$
b^{[l]}\in\mathbb{R}^{n_l}
$$

There is one bias per neuron in the layer.

## 10.4 Vectorized Calculation

The layer's pre-activation is:

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

The activation is:

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

For a fully connected network, these equations are repeated for each
trainable layer.

## 10.5 Why Is Matrix Multiplication Useful?

Suppose a layer has 128 neurons and receives 256 values. Calculating
each neuron independently would require many separate dot products.
Matrix multiplication calculates all 128 weighted sums together,
allowing optimized numerical libraries and hardware to operate
efficiently.

------------------------------------------------------------------------

# 11. Understanding Matrix and Vector Dimensions

Dimension errors are among the most common neural-network implementation
problems.

Suppose: - Input features: (n_0=3). - Hidden neurons: (n_1=4). - Output
neurons: (n_2=2).

Then:

$$
a^{[0]}\in\mathbb{R}^{3}
$$

$$
W^{[1]}\in\mathbb{R}^{4\times 3},
\qquad b^{[1]}\in\mathbb{R}^{4}
$$

The multiplication is valid:

$$
(4\times 3)(3\times 1)=(4\times 1)
$$

Therefore:

$$
z^{[1]}\in\mathbb{R}^{4},
\qquad a^{[1]}\in\mathbb{R}^{4}
$$

For the output layer:

$$
W^{[2]}\in\mathbb{R}^{2\times 4},
\qquad b^{[2]}\in\mathbb{R}^{2}
$$

And:

$$
(2\times 4)(4\times 1)=(2\times 1)
$$

So:

$$
z^{[2]},a^{[2]}\in\mathbb{R}^{2}
$$

## 11.1 The Inner-Dimension Rule

For matrix multiplication:

$$
(A_{p\times q})(B_{r\times s})
$$

the operation is valid only when (q=r). The result has shape
(p`\times `{=tex}s).

In a neural network, the number of columns in the weight matrix must
match the number of activations from the previous layer.

## 11.2 Bias Broadcasting

In a single-example vector calculation, the bias vector has one value
per neuron. In batch calculations, libraries typically broadcast the
bias across all examples.

For example, if:

$$
Z^{[l]}\in\mathbb{R}^{m\times n_l}
$$

and:

$$
b^{[l]}\in\mathbb{R}^{n_l}
$$

then the bias is added to each of the (m) rows.

------------------------------------------------------------------------

# 12. Complete Numerical Example: One Hidden Layer

Let's calculate a small network completely by hand.

## 12.1 Network Setup

Architecture:

$$
2 \rightarrow 2 \rightarrow 1
$$

Input:

$$
x=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

Hidden-layer weights and bias:

$$
W^{[1]}=
\begin{bmatrix}
0.1 & 0.2\\
0.3 & 0.4
\end{bmatrix}
$$

$$
b^{[1]}=
\begin{bmatrix}
0.1\\
0.1
\end{bmatrix}
$$

Output-layer weights and bias:

$$
W^{[2]}=
\begin{bmatrix}
0.5 & -0.2
\end{bmatrix}
$$

$$
b^{[2]}=
\begin{bmatrix}
0.1
\end{bmatrix}
$$

Use ReLU in the hidden layer and sigmoid in the output layer.

## 12.2 Step 1: Calculate Hidden Pre-Activations

$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$

First, multiply the matrix and input:

$$
W^{[1]}x=
\begin{bmatrix}
0.1 & 0.2\\
0.3 & 0.4
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

First row:

$$
(0.1)(1)+(0.2)(2)=0.1+0.4=0.5
$$

Second row:

$$
(0.3)(1)+(0.4)(2)=0.3+0.8=1.1
$$

Therefore:

$$
W^{[1]}x=
\begin{bmatrix}
0.5\\
1.1
\end{bmatrix}
$$

Add the bias:

$$
z^{[1]}=
\begin{bmatrix}
0.5\\
1.1
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
0.1
\end{bmatrix}
=
\begin{bmatrix}
0.6\\
1.2
\end{bmatrix}
$$

## 12.3 Step 2: Apply ReLU

$$
a^{[1]}=\operatorname{ReLU}(z^{[1]})
$$

Both values are positive, so ReLU leaves them unchanged:

$$
a^{[1]}=
\begin{bmatrix}
0.6\\
1.2
\end{bmatrix}
$$

This vector becomes the input to the output layer.

## 12.4 Step 3: Calculate the Output Pre-Activation

$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$

Substitute:

$$
z^{[2]}=
\begin{bmatrix}
0.5 & -0.2
\end{bmatrix}
\begin{bmatrix}
0.6\\
1.2
\end{bmatrix}
+0.1
$$

Calculate the dot product:

$$
z^{[2]}=(0.5)(0.6)+(-0.2)(1.2)+0.1
$$

$$
z^{[2]}=0.3-0.24+0.1=0.16
$$

## 12.5 Step 4: Apply Sigmoid

$$
\hat{y}=\sigma(0.16)=\frac{1}{1+e^{-0.16}}
$$

Numerically:

$$
\hat{y}\approx 0.5399
$$

The model's output is approximately (0.54).

If this is a binary classifier and the chosen decision threshold is
(0.5), the predicted class would be 1. The probability-like output and
the final class decision are distinct: changing the threshold can change
the class without changing the network's forward-pass output.

## 12.6 Full Calculation in One Expression

$$
\boxed{
\hat{y}=
\sigma\left(
W^{[2]}\operatorname{ReLU}(W^{[1]}x+b^{[1]})+b^{[2]}
\right)
}
$$

For the chosen numbers:

$$
\boxed{\hat{y}\approx 0.5399}
$$

## 12.7 Verify With NumPy

``` python
import numpy as np

x = np.array([[1.0],
              [2.0]])

W1 = np.array([[0.1, 0.2],
               [0.3, 0.4]])
b1 = np.array([[0.1],
               [0.1]])

W2 = np.array([[0.5, -0.2]])
b2 = np.array([[0.1]])

# Hidden layer
z1 = W1 @ x + b1
a1 = np.maximum(0, z1)  # ReLU

# Output layer
z2 = W2 @ a1 + b2
y_hat = 1 / (1 + np.exp(-z2))  # Sigmoid

print("z1:\n", z1)
print("a1:\n", a1)
print("z2:\n", z2)
print("Prediction:\n", y_hat)
```

Expected approximate output:

``` text
z1:
[[0.6]
 [1.2]]
a1:
[[0.6]
 [1.2]]
z2:
[[0.16]]
Prediction:
[[0.5399]]
```

------------------------------------------------------------------------

# 13. Forward Propagation for Multiple Examples (Batch Processing)

Training and inference often process many examples at once. Instead of
computing each example separately, we arrange the examples into a
matrix.

## 13.1 Batch Input Convention

In this section, use the common machine-learning convention that each
row represents one example.

If there are (m) examples and (n_0) features:

$$
X\in\mathbb{R}^{m\times n_0}
$$

For example, 5 examples with 3 features each:

$$
X\in\mathbb{R}^{5\times 3}
$$

The first dimension is the batch size, and the second is the number of
features.

## 13.2 Batch Formula

For a layer with (n_l) neurons:

$$
W^{[l]}\in\mathbb{R}^{n_{l-1}\times n_l}
$$

Under the row-per-example convention, the batch formula is:

$$
Z^{[l]}=A^{[l-1]}W^{[l]}+b^{[l]}
$$

where:

$$
A^{[0]}=X
$$

The resulting matrix has shape:

$$
Z^{[l]}\in\mathbb{R}^{m\times n_l}
$$

Then apply the activation elementwise:

$$
A^{[l]}=g^{[l]}(Z^{[l]})
$$

**Important:** This batch formula stores the weight matrix in the
transpose orientation compared with the column-vector convention used in
Sections 9--12. Both conventions are correct; do not mix them in one
calculation.

## 13.3 Batch Numerical Example

Suppose there are three examples and two features:

$$
X=
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6
\end{bmatrix}
$$

Let a layer have two neurons:

$$
W=
\begin{bmatrix}
0.1 & 0.3\\
0.2 & 0.4
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
0.1 & 0.1
\end{bmatrix}
$$

Then:

$$
Z=XW+b
$$

For the first example (\[1,2\]):

$$
[1,2]W=[0.5,1.1]
$$

After adding the bias:

$$
[0.6,1.2]
$$

For the second example (\[3,4\]):

$$
[3,4]W=[1.1,2.5]
$$

After adding the bias:

$$
[1.2,2.6]
$$

For the third example (\[5,6\]):

$$
[5,6]W=[1.7,3.9]
$$

After adding the bias:

$$
[1.8,4.0]
$$

So:

$$
Z=
\begin{bmatrix}
0.6 & 1.2\\
1.2 & 2.6\\
1.8 & 4.0
\end{bmatrix}
$$

With ReLU, these values remain unchanged because all are positive.

## 13.4 Why Batches Are Efficient

Batch processing: - Reduces Python-level loops. - Uses optimized matrix
multiplication. - Can make better use of CPUs, GPUs, and accelerators. -
Produces a prediction for each example in one vectorized operation.

The batch size does not change the mathematical function applied to an
individual example; it changes how many examples are computed together.

------------------------------------------------------------------------

# 14. Forward Propagation for Binary Classification

Binary classification predicts one of two classes, such as
positive/negative or spam/not spam.

## 14.1 Output Logit

A network first calculates a real-valued logit:

$$
z^{[L]}=W^{[L]}a^{[L-1]}+b^{[L]}
$$

The logit can be any real number.

## 14.2 Sigmoid Output

$$
\hat{y}=\sigma(z^{[L]})=\frac{1}{1+e^{-z^{[L]}}}
$$

For finite logits, (`\hat{y}`{=tex}) lies strictly between 0 and 1.

Example:

$$
z^{[L]}=2
$$

$$
\hat{y}=\sigma(2)\approx 0.8808
$$

The network assigns approximately 0.88 probability to the positive
class, subject to the model's quality and calibration.

## 14.3 Thresholding

A common decision rule is:

$$
\hat{c}=
\begin{cases}
1, & \hat{y}\geq 0.5\\
0, & \hat{y}<0.5
\end{cases}
$$

This threshold is not universal. In fraud detection, medical screening,
and other cost-sensitive tasks, a different threshold may be more
appropriate.

## 14.4 Binary Cross-Entropy (Context)

Forward propagation produces (`\hat{y}`{=tex}). During supervised
training, the prediction is compared with the target (y) using a loss
function such as binary cross-entropy:

$$
\mathcal{L}(y,\hat{y})=
-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]
$$

For (m) examples, the mean loss is:

$$
J=-\frac{1}{m}\sum_{i=1}^{m}
\left[
y^{(i)}\log\hat{y}^{(i)}
+(1-y^{(i)})\log(1-\hat{y}^{(i)})
\right]
$$

The loss calculation is a step after forward propagation, not the same
operation as the forward pass.

In practical code, sigmoid and binary cross-entropy are often combined
in a numerically stable "with logits" loss. In that case, the model may
return logits and the loss function handles the sigmoid internally.

------------------------------------------------------------------------

# 15. Forward Propagation for Multiclass Classification and Softmax

Suppose a model classifies an image into one of three mutually exclusive
classes: cat, dog, or bird.

The final layer has three output neurons:

$$
z=
\begin{bmatrix}
z_1\\
z_2\\
z_3
\end{bmatrix}
$$

These raw outputs are called **logits**.

## 15.1 Softmax Calculation

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Suppose:

$$
z=
\begin{bmatrix}
2\\
1\\
0
\end{bmatrix}
$$

Exponentials are approximately:

$$
e^2\approx 7.389,\qquad e^1\approx 2.718,\qquad e^0=1
$$

Their sum is:

$$
7.389+2.718+1=11.107
$$

Therefore:

$$
p_1\approx\frac{7.389}{11.107}\approx 0.665
$$

$$
p_2\approx\frac{2.718}{11.107}\approx 0.245
$$

$$
p_3\approx\frac{1}{11.107}\approx 0.090
$$

The probability vector is approximately:

$$
p=
\begin{bmatrix}
0.665\\
0.245\\
0.090
\end{bmatrix}
$$

The entries sum to approximately 1, with small differences possible due
to rounding.

## 15.2 Class Decision

The predicted class is typically the index of the largest probability:

$$
\hat{c}=\arg\max_i p_i
$$

In the example, class 1 has the largest probability.

## 15.3 Categorical Cross-Entropy

For a one-hot target vector (y):

$$
\mathcal{L}(y,p)=-\sum_{i=1}^{K}y_i\log p_i
$$

If the correct class is class 1, then (y=\[1,0,0\]\^T), so:

$$
\mathcal{L}=-\log(p_1)
$$

As the predicted probability for the correct class increases, this loss
decreases.

------------------------------------------------------------------------

# 16. Forward Propagation for Regression

Regression predicts a continuous numerical value, such as a house price
or temperature.

## 16.1 Linear Output

For a regression output, a common choice is no output activation:

$$
\hat{y}=z^{[L]}
$$

For a single output neuron:

$$
\hat{y}=w^Ta+b
$$

The output can be any real number.

## 16.2 Example

Suppose the last hidden representation is:

$$
a=
\begin{bmatrix}
2\\
1
\end{bmatrix}
$$

The output weights are:

$$
w=
\begin{bmatrix}
3\\
-0.5
\end{bmatrix},
\qquad b=4
$$

Then:

$$
\hat{y}=3(2)-0.5(1)+4=9.5
$$

The network predicts (9.5) in the target's units.

## 16.3 Mean Squared Error (Context)

After the forward pass, regression predictions may be evaluated using
mean squared error:

$$
\operatorname{MSE}=\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-\hat{y}^{(i)})^2
$$

The forward pass computes predictions; MSE measures their squared
errors.

------------------------------------------------------------------------

# 17. Forward Propagation in Deep Neural Networks

A deep neural network contains multiple hidden layers. The same layer
formula is repeated.

For a network with (L) trainable layers:

$$
a^{[0]}=x
$$

For each layer (l=1,`\ldots`{=tex},L):

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

Finally:

$$
\hat{y}=a^{[L]}
$$

## 17.1 Example: Four Trainable Layers

Suppose the architecture is:

$$
5 \rightarrow 8 \rightarrow 6 \rightarrow 4 \rightarrow 2
$$

Then: - Input: 5 features. - Layer 1: 8 neurons. - Layer 2: 6 neurons. -
Layer 3: 4 neurons. - Layer 4: 2 output neurons.

The calculations are:

$$
a^{[1]}=g^{[1]}(W^{[1]}x+b^{[1]})
$$

$$
a^{[2]}=g^{[2]}(W^{[2]}a^{[1]}+b^{[2]})
$$

$$
a^{[3]}=g^{[3]}(W^{[3]}a^{[2]}+b^{[3]})
$$

$$
a^{[4]}=g^{[4]}(W^{[4]}a^{[3]}+b^{[4]})
$$

If the final layer is a two-class softmax layer, the last activation
vector contains two class probabilities.

## 17.2 Why Use Multiple Layers?

Multiple layers allow the network to build successive representations.
In image models, early layers may learn local patterns and deeper layers
may combine them into more complex patterns. This is a broad intuition
rather than a guarantee that every individual neuron has a simple
human-interpretable meaning.

## 17.3 The Dependency Chain

The computation is sequential across layers:

$$
a^{[1]}\rightarrow z^{[2]}\rightarrow a^{[2]}\rightarrow\cdots\rightarrow a^{[L]}
$$

Layer (l) cannot compute its ordinary forward activation until the
previous layer's activation is available.

------------------------------------------------------------------------

# 18. Forward Propagation in Tensor Form

Real neural networks often process tensors rather than simple vectors.

## 18.1 What Is a Tensor?

A tensor generalizes scalars, vectors, and matrices:

-   Scalar: rank 0.
-   Vector: rank 1.
-   Matrix: rank 2.
-   Higher-dimensional array: rank 3 or more.

Examples: - Tabular batch: ((m,`\text{features}`{=tex})). - Grayscale
image batch: ((m,h,w,1)) in channels-last format. - RGB image batch:
((m,h,w,3)) in channels-last format. - Token IDs: often
((m,`\text{sequence length}`{=tex})). - Token embeddings: often
((m,`\text{sequence length}`{=tex},`\text{embedding dimension}`{=tex})).

Exact layouts depend on the library and model.

## 18.2 Dense Layers

A dense layer applies an affine transformation to the final feature
dimension, then an activation:

$$
Y=XW+b
$$

For a batch of vectors, the leading dimensions may represent batch or
other grouping dimensions, while the final dimension is the feature
dimension.

## 18.3 Convolutional Neural Networks

For a convolutional layer, forward propagation applies learned filters
over local spatial regions, often adds a bias, and applies an
activation. It is not usually expressed as one ordinary dense matrix
multiplication over the full image, though convolution can be
represented as a structured linear operation.

## 18.4 Transformer Networks

In a Transformer, forward propagation includes operations such as: -
Token embedding and positional information. - Linear projections for
queries, keys, and values. - Attention-score calculation and weighted
value aggregation. - Feed-forward transformations. - Residual
connections and normalization.

These operations form a larger computation graph, but the same general
idea applies: input tensors are transformed through a sequence of
operations to produce output tensors.

------------------------------------------------------------------------

# 19. NumPy Implementation From Scratch

This section implements a small dense neural network using NumPy.

## 19.1 Define the Activation Functions

``` python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    # Clip only to reduce overflow in this simple educational implementation.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

For production code, prefer a well-tested framework implementation,
especially for numerically sensitive operations.

## 19.2 Implement a Forward-Pass Function

This version uses the column-vector convention: one example has shape
`(n_features, 1)`, weights have shape `(n_neurons, n_previous)`, and
biases have shape `(n_neurons, 1)`.

``` python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def forward_propagation(x, W1, b1, W2, b2):
    # Hidden layer
    z1 = W1 @ x + b1
    a1 = relu(z1)

    # Output layer
    z2 = W2 @ a1 + b2
    y_hat = sigmoid(z2)

    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "y_hat": y_hat,
    }
    return y_hat, cache
```

## 19.3 Run the Example

``` python
x = np.array([[1.0],
              [2.0]])

W1 = np.array([[0.1, 0.2],
               [0.3, 0.4]])
b1 = np.array([[0.1],
               [0.1]])

W2 = np.array([[0.5, -0.2]])
b2 = np.array([[0.1]])

prediction, cache = forward_propagation(x, W1, b1, W2, b2)

print("Hidden pre-activation:\n", cache["z1"])
print("Hidden activation:\n", cache["a1"])
print("Output logit:\n", cache["z2"])
print("Prediction:\n", prediction)
```

Expected prediction:

``` text
[[0.5399]]
```

The `cache` stores intermediate values. During training, gradient
calculations often need these values, which is why frameworks retain or
recompute relevant intermediate tensors.

## 19.4 Batch Version

This version uses one row per example:

``` python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def forward_batch(X, W1, b1, W2, b2):
    # X: (batch_size, input_features)
    # W1: (input_features, hidden_units)
    # b1: (hidden_units,)
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # W2: (hidden_units, output_units)
    # b2: (output_units,)
    Z2 = A1 @ W2 + b2
    Y_hat = sigmoid(Z2)

    return Y_hat, {"Z1": Z1, "A1": A1, "Z2": Z2}
```

Example:

``` python
X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

W1 = np.array([
    [0.1, 0.3],
    [0.2, 0.4],
])
b1 = np.array([0.1, 0.1])

W2 = np.array([
    [0.5],
    [-0.2],
])
b2 = np.array([0.1])

predictions, cache = forward_batch(X, W1, b1, W2, b2)
print(predictions.shape)  # (3, 1)
print(predictions)
```

The output has one prediction per example, so its shape is `(3, 1)`.

## 19.5 Generalizing to Any Number of Layers

``` python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def forward_deep(x, weights, biases, hidden_activation=relu,
                 output_activation=sigmoid):
    # Column-vector convention:
    # x has shape (input_features, 1)
    # weights[l] has shape (neurons_l, neurons_previous)
    # biases[l] has shape (neurons_l, 1)
    a = x
    activations = [x]
    pre_activations = []

    number_of_layers = len(weights)

    for layer_index in range(number_of_layers):
        z = weights[layer_index] @ a + biases[layer_index]
        pre_activations.append(z)

        if layer_index == number_of_layers - 1:
            a = output_activation(z)
        else:
            a = hidden_activation(z)

        activations.append(a)

    cache = {
        "pre_activations": pre_activations,
        "activations": activations,
    }
    return a, cache
```

This educational function assumes every hidden layer uses the same
activation and the final layer uses the specified output activation.
More complex networks may use different activation functions, residual
connections, normalization layers, dropout, or other operations.

------------------------------------------------------------------------

# 20. Forward Propagation Using Keras

A high-level framework handles tensor operations and automatic
differentiation.

## 20.1 Build a Simple Binary Classifier

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(2, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.summary()
```

The model has: - 2 input features. - 2 hidden units with ReLU. - 1
output unit with sigmoid.

## 20.2 Inspect the Prediction

``` python
import numpy as np

x = np.array([[1.0, 2.0]], dtype=np.float32)
prediction = model(x, training=False)

print(prediction.numpy())
```

The network performs forward propagation when `model(x, training=False)`
is called. Because the model has not been trained and its parameters are
initialized automatically, the prediction is not expected to be
meaningful yet.

## 20.3 Compile and Train

``` python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# X_train must have shape (number_of_examples, 2).
# y_train must contain binary targets, usually 0 or 1.
# model.fit(X_train, y_train, epochs=10, batch_size=32)
```

The `fit` process repeatedly performs forward passes, evaluates the
loss, computes gradients, and updates trainable parameters.

## 20.4 Get Intermediate Activations

``` python
intermediate_model = keras.Model(
    inputs=model.inputs,
    outputs=[
        model.layers[0].output,
        model.layers[1].output,
    ],
)

x = np.array([[1.0, 2.0]], dtype=np.float32)
hidden_activation, prediction = intermediate_model(x, training=False)

print("Hidden activation:", hidden_activation.numpy())
print("Prediction:", prediction.numpy())
```

Intermediate activations are useful for understanding what each layer
computes and for debugging shape or numerical issues.

------------------------------------------------------------------------

# 21. Forward Propagation During Training vs. Inference

The mathematical core of a dense layer is the same in training and
inference, but some layers behave differently depending on the mode.

## 21.1 Training

During training: 1. A batch of inputs enters the network. 2. Forward
propagation calculates predictions. 3. A loss function compares
predictions with targets. 4. Backpropagation calculates gradients. 5. An
optimizer updates parameters.

A simplified iteration is:

$$
\hat{y}=f_{\theta}(X)
$$

$$
J=\mathcal{L}(y,\hat{y})
$$

$$
\theta\leftarrow\operatorname{OptimizerUpdate}(\theta,\nabla_{\theta}J)
$$

The final line represents an optimizer-dependent update, not a single
universal formula.

## 21.2 Inference

During inference, the model uses its learned parameters to produce
predictions. Usually there is no loss-based parameter update:

$$
\hat{y}=f_{\theta}(X)
$$

In frameworks such as Keras, `training=False` explicitly requests
inference behavior for layers that distinguish modes.

## 21.3 Dropout

Dropout randomly disables a fraction of activations during training to
reduce over-reliance on particular units. It is generally disabled
during inference, with framework-specific scaling ensuring the expected
activation scale is handled appropriately.

## 21.4 Batch Normalization

Batch normalization uses batch statistics during training and moving
statistics during inference in its common configuration. Therefore, its
behavior can differ between training and inference even though both are
forward computations.

## 21.5 Important Distinction

Forward propagation is the calculation of the output. Training is the
broader process that uses the forward pass, loss, gradients, and
optimizer updates to improve parameters.

------------------------------------------------------------------------

# 22. Common Mistakes and Debugging Checklist

## 22.1 Forgetting the Bias

Incorrect:

``` python
z = W @ x
```

If the layer has a bias, the usual affine transformation is:

``` python
z = W @ x + b
```

A bias may be intentionally omitted in some architectures, but it should
not be forgotten accidentally.

## 22.2 Applying the Activation at the Wrong Time

The usual dense-layer order is:

$$
z=Wa+b
$$

$$
a_{\text{new}}=g(z)
$$

Applying the activation before the weighted sum changes the model.

## 22.3 Mixing Up (z) and (a)

-   (z): before activation.
-   (a): after activation.

The next layer normally receives the previous layer's activation.

## 22.4 Shape Mismatch

If the input has 3 features and the layer has 4 neurons, a column-vector
weight matrix should have shape `(4, 3)`. In the row-per-example
convention, the corresponding weights should have shape `(3, 4)`.

Check shapes explicitly:

``` python
print("X:", X.shape)
print("W1:", W1.shape)
print("b1:", b1.shape)
print("Z1:", Z1.shape)
```

## 22.5 Confusing Logits With Probabilities

A logit is an unconstrained real number. A sigmoid or softmax activation
transforms logits into probability-like outputs.

Do not apply sigmoid twice. If a loss expects logits, pass logits to
that loss rather than applying the sigmoid manually first.

## 22.6 Wrong Output Activation

Typical combinations:

-   Binary classification: sigmoid with binary cross-entropy, or logits
    with a logits-aware binary cross-entropy.
-   Single-label multiclass classification: softmax with
    categorical/sparse categorical cross-entropy, or logits with a
    logits-aware loss.
-   Regression: commonly a linear output with an appropriate regression
    loss.

Always check the framework loss configuration and whether it expects
logits or probabilities.

## 22.7 Numerical Overflow in Sigmoid or Softmax

Directly evaluating (e\^{-z}) or (e\^z) for extreme values can overflow.
Stable library implementations are preferable for production. For
softmax, subtracting the maximum logit does not change the result:

$$
\operatorname{softmax}(z)_i=
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
$$

where:

$$
c=\max_j z_j
$$

## 22.8 Assuming an Untrained Model Predicts Meaningfully

A newly initialized model can run a forward pass, but its predictions
are generally not useful until it has been trained (or its weights have
otherwise been set appropriately).

## 22.9 Confusing Prediction With Decision

A probability such as (0.54) is not automatically a final business
decision. A threshold, cost function, or policy may be needed to convert
predictions into actions.

## 22.10 Debugging Checklist

-   [ ] Does the input have the expected shape?
-   [ ] Do the weight matrix dimensions match the previous layer?
-   [ ] Is the bias shape compatible?
-   [ ] Is the weighted sum calculated before activation?
-   [ ] Does each layer receive the previous layer's activation?
-   [ ] Is the final output shape appropriate for the task?
-   [ ] Is the output activation compatible with the loss?
-   [ ] Are logits and probabilities being handled consistently?
-   [ ] Are predictions finite (`NaN` and infinity absent)?
-   [ ] Is inference mode selected when appropriate?

------------------------------------------------------------------------

# 23. Computational Cost and Numerical Stability

## 23.1 Approximate Cost of a Dense Layer

A dense layer with (n\_{l-1}) inputs and (n_l) neurons performs
approximately:

$$
n_{l-1}n_l
$$

weight multiplications per example, plus additions and activation
calculations.

For a batch of (m) examples, the main matrix multiplication costs on the
order of:

$$
O(mn_{l-1}n_l)
$$

This is a rough operation-count estimate; actual runtime depends on
hardware, data types, memory movement, and library implementation.

## 23.2 Number of Parameters

For a dense layer with a bias per neuron:

$$
\text{Parameters}=n_{l-1}n_l+n_l
$$

Equivalently:

$$
\text{Parameters}=(n_{l-1}+1)n_l
$$

For a network, sum the parameter count across its trainable layers.

Example: 3 input features and 4 neurons:

$$
3(4)+4=16
$$

There are 12 weights and 4 biases, giving 16 trainable parameters.

## 23.3 Numerical Stability

Large logits can cause overflow in naïve exponential calculations. Deep
networks may also produce very large or very small intermediate values.
Stable implementations, sensible initialization, normalization where
appropriate, and monitoring activations help prevent numerical problems.

## 23.4 Why Save Intermediate Values?

The forward pass can store intermediate activations and pre-activations.
Backpropagation often needs these values to calculate derivatives
efficiently. Frameworks manage this bookkeeping automatically, sometimes
recomputing certain values to reduce memory use.

------------------------------------------------------------------------

# 24. Common Interview Questions

## Q1. What is forward propagation?

Forward propagation passes inputs through the network's layers to
compute intermediate activations and the final prediction.

## Q2. What is the basic equation of a dense layer?

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

Then the activation is:

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

## Q3. What is the difference between weights and biases?

Weights scale input contributions. Biases add an adjustable offset to
the pre-activation.

## Q4. What is the difference between pre-activation and activation?

Pre-activation (z) is the weighted sum plus bias. Activation (a) is the
result after applying the activation function.

## Q5. Does forward propagation update weights?

No. It calculates outputs using the current weights and biases.
Parameter updates occur during training through an optimizer using
gradients.

## Q6. What is the difference between forward propagation and backpropagation?

Forward propagation computes predictions. Backpropagation computes
derivatives of the loss with respect to intermediate values and
parameters by applying the chain rule backward through the computation
graph.

## Q7. Why are activation functions used?

Nonlinear activations allow stacked layers to represent nonlinear
functions. Without nonlinearities, a stack of affine layers collapses to
a single affine transformation.

## Q8. Why is ReLU commonly used in hidden layers?

It is computationally simple and does not saturate on the positive side.
However, units can become inactive when their inputs remain negative.

## Q9. Why is sigmoid used for binary classification?

It maps one logit to a value between 0 and 1, which can be interpreted
as the positive-class probability under the model.

## Q10. Why is softmax used for multiclass classification?

It transforms a vector of logits into positive values that sum to 1,
making it suitable for a distribution over mutually exclusive classes.

## Q11. What is a logit?

A logit is the raw, unnormalized score produced before a sigmoid or
softmax transformation.

## Q12. What is the shape of the weight matrix?

Under the column-vector convention:

$$
W^{[l]}\in\mathbb{R}^{n_l\times n_{l-1}}
$$

Under the row-per-example batch convention:

$$
W^{[l]}\in\mathbb{R}^{n_{l-1}\times n_l}
$$

Always state the convention being used.

## Q13. Why do we use matrix multiplication?

It computes many weighted sums efficiently and takes advantage of
optimized numerical libraries and accelerators.

## Q14. What is batch forward propagation?

It calculates outputs for multiple examples together by arranging the
examples into a matrix and using vectorized operations.

## Q15. What is the role of the output layer?

It transforms the final hidden representation into task-appropriate
outputs, such as a continuous value, binary probability, or multiclass
probability distribution.

## Q16. Can a model perform forward propagation before training?

Yes. A newly initialized model can produce outputs, but those outputs
are generally not meaningful until the model is trained or its
parameters are set appropriately.

## Q17. What happens if every layer uses only linear transformations?

The composition is still linear/affine, so additional layers do not
provide the expressive benefit of nonlinear hidden representations.

## Q18. What is broadcasting?

Broadcasting is a tensor-library mechanism that allows a bias vector to
be added across multiple examples without manually copying the vector
into every row.

## Q19. What is the difference between training and inference forward passes?

Some layers, such as dropout and batch normalization, behave differently
depending on the mode. Inference usually disables dropout and uses
inference statistics for batch normalization.

## Q20. What is the relationship between forward propagation and loss?

Forward propagation produces predictions. A loss function then compares
those predictions with targets during supervised training.

------------------------------------------------------------------------

# 25. Practice Problems

Try solving these before checking the answers.

## Problem 1: Single Neuron

Given:

$$
x_1=2,\quad x_2=4
$$

$$
w_1=0.5,\quad w_2=0.25,\quad b=-1
$$

Calculate (z), then calculate the ReLU activation.

## Problem 2: Sigmoid

Calculate:

$$
\sigma(0)
$$

What does this value mean for a binary-classification output?

## Problem 3: Matrix Dimensions

A layer receives 6 features and contains 10 neurons.

1.  What is the weight shape under the column-vector convention?
2.  What is the bias shape?
3.  What is the output shape for one example?

## Problem 4: Parameter Count

A dense layer receives 8 features and contains 5 neurons. How many
trainable parameters does it have if every neuron has a bias?

## Problem 5: Hidden Layer

Given:

$$
x=
\begin{bmatrix}
1\\
2
\end{bmatrix}
,\quad
W=
\begin{bmatrix}
1 & -1\\
0.5 & 2
\end{bmatrix}
,\quad
b=
\begin{bmatrix}
0\\
-1
\end{bmatrix}
$$

Calculate (z=Wx+b) and then apply ReLU.

## Problem 6: Output Layer

Suppose the hidden activation is:

$$
a=
\begin{bmatrix}
2\\
1
\end{bmatrix}
$$

and:

$$
W=
\begin{bmatrix}
0.5 & 1
\end{bmatrix},
\qquad b=-0.5
$$

Calculate the logit. If sigmoid is used, calculate the approximate
output.

## Problem 7: Softmax

Given logits:

$$
z=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

Calculate the softmax probabilities.

## Problem 8: Batch Shape

There are 100 examples and each has 12 features. What is the shape of
the input matrix using the row-per-example convention?

## Problem 9: Forward Propagation vs. Training

Explain why computing a prediction does not automatically update the
model's weights.

## Problem 10: Activation Function

Why does stacking multiple linear layers without nonlinear activations
fail to provide the expressive power expected from a deep nonlinear
network?

------------------------------------------------------------------------

## Answer Key

### Answer 1

$$
z=(0.5)(2)+(0.25)(4)-1=1+1-1=1
$$

$$
\operatorname{ReLU}(1)=1
$$

### Answer 2

$$
\sigma(0)=\frac{1}{1+e^0}=\frac{1}{2}=0.5
$$

The logit is exactly zero, so the sigmoid output is 0.5.

### Answer 3

Under the column-vector convention:

$$
W\in\mathbb{R}^{10\times 6},\quad b\in\mathbb{R}^{10\times 1}
$$

For one example, the output has shape (10`\times 1`{=tex}).

### Answer 4

$$
8(5)+5=40+5=45
$$

There are 40 weights and 5 biases.

### Answer 5

$$
Wx=
\begin{bmatrix}
1 & -1\\
0.5 & 2
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
-1\\
4.5
\end{bmatrix}
$$

Add the bias:

$$
z=
\begin{bmatrix}
-1\\
4.5
\end{bmatrix}
+
\begin{bmatrix}
0\\
-1
\end{bmatrix}
=
\begin{bmatrix}
-1\\
3.5
\end{bmatrix}
$$

Apply ReLU:

$$
a=
\begin{bmatrix}
0\\
3.5
\end{bmatrix}
$$

### Answer 6

$$
z=(0.5)(2)+(1)(1)-0.5=1+1-0.5=1.5
$$

$$
\sigma(1.5)\approx 0.8176
$$

### Answer 7

Both logits are equal, so the probabilities are equal:

$$
p_1=p_2=0.5
$$

### Answer 8

$$
X\in\mathbb{R}^{100\times 12}
$$

In NumPy, its shape is `(100, 12)`.

### Answer 9

Forward propagation only evaluates the network using the current
parameters. Updating parameters requires calculating a loss, computing
gradients, and applying an optimizer update.

### Answer 10

The composition of affine transformations is itself an affine
transformation. Nonlinear activation functions are what allow stacked
layers to represent richer nonlinear mappings.

------------------------------------------------------------------------

# 26. Summary and Key Formulas

## 26.1 The Forward Propagation Pipeline

$$
\boxed{
\text{Input}
\rightarrow
\text{Weighted Sum + Bias}
\rightarrow
\text{Activation}
\rightarrow
\text{Next Layer}
\rightarrow
\text{Prediction}
}
$$

## 26.2 Single Neuron

$$
\boxed{z=w^Tx+b}
$$

$$
\boxed{a=g(z)}
$$

Combined:

$$
\boxed{a=g(w^Tx+b)}
$$

## 26.3 One Layer

Using the column-vector convention:

$$
\boxed{z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}}
$$

$$
\boxed{a^{[l]}=g^{[l]}(z^{[l]})}
$$

## 26.4 Whole Network

$$
\boxed{a^{[0]}=x}
$$

$$
\boxed{
a^{[l]}=g^{[l]}\left(W^{[l]}a^{[l-1]}+b^{[l]}\right)
}
$$

for (l=1,`\ldots`{=tex},L).

The final prediction is:

$$
\boxed{\hat{y}=a^{[L]}}
$$

## 26.5 Batch Formula

Using one example per row:

$$
\boxed{Z^{[l]}=A^{[l-1]}W^{[l]}+b^{[l]}}
$$

$$
\boxed{A^{[l]}=g^{[l]}(Z^{[l]})}
$$

## 26.6 Common Output Functions

Sigmoid:

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

ReLU:

$$
\boxed{\operatorname{ReLU}(z)=\max(0,z)}
$$

Softmax:

$$
\boxed{
\operatorname{softmax}(z)_i=
\frac{e^{z_i}}{\sum_j e^{z_j}}
}
$$

## 26.7 Dense Layer Parameter Count

$$
\boxed{\text{Parameters}=n_{\text{in}}n_{\text{out}}+n_{\text{out}}}
$$

## 26.8 Final Takeaways

1.  Forward propagation computes a neural network's output from its
    input and current parameters.
2.  Each dense layer calculates a weighted sum, adds a bias, and applies
    an activation function.
3.  The activation of one layer becomes the input to the next layer.
4.  Matrix multiplication makes the computation efficient and
    vectorized.
5.  Matrix dimensions must align with the number of inputs and neurons.
6.  Batch processing computes predictions for many examples at once.
7.  The output activation depends on the task: sigmoid for common binary
    classification, softmax for mutually exclusive multiclass
    classification, and often a linear output for regression.
8.  Forward propagation alone does not train the network or update its
    parameters.
9.  During training, the forward pass is followed by loss calculation,
    backpropagation, and an optimizer update.
10. Correct notation, tensor shapes, and a clear distinction between
    logits and activations make neural-network code much easier to
    understand and debug.

------------------------------------------------------------------------

**Study suggestion:** First work through Sections 6--12 by hand, then
run the NumPy example in Section 19, and finally inspect the Keras model
in Section 20. If these three stages make sense, you have a strong
practical and mathematical foundation for forward propagation.
