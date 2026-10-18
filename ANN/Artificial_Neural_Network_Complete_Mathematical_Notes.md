# Artificial Neural Network (ANN): Complete Mathematical and Conceptual Notes

## Beginner to Advanced \| Architecture \| Mathematics \| Forward Propagation \| Backpropagation \| NumPy \| TensorFlow/Keras

------------------------------------------------------------------------

## Table of Contents

1.  Introduction to Artificial Neural Networks
2.  What Problems Can ANN Solve?
3.  Biological Neuron vs. Artificial Neuron
4.  Structure of an Artificial Neural Network
5.  The Artificial Neuron: Mathematical Model
6.  Weights, Biases, Inputs, and Activations
7.  Activation Functions in ANN
8.  ANN Architecture and Layer Types
9.  Forward Propagation
10. Loss Functions
11. Backpropagation: The Core Idea
12. Derivatives and the Chain Rule
13. Backpropagation in a Single Neuron
14. Backpropagation in a Two-Layer Network
15. Gradient Descent and Parameter Updates
16. Complete Training Loop
17. Weight Initialization
18. Learning Rate, Epochs, Batches, and Iterations
19. Overfitting, Underfitting, and Generalization
20. Regularization Techniques
21. Optimizers: SGD, Momentum, RMSprop, and Adam
22. ANN for Regression
23. ANN for Binary Classification
24. ANN for Multiclass Classification
25. NumPy ANN From Scratch
26. ANN Using TensorFlow/Keras
27. Evaluating an ANN
28. Common ANN Problems and Debugging
29. ANN vs. Traditional Machine Learning
30. ANN vs. CNN vs. RNN vs. Transformer
31. Practical Applications
32. Interview Questions and Answers
33. Practice Problems
34. Answer Key
35. Summary and Formula Sheet

------------------------------------------------------------------------

# 1. Introduction to Artificial Neural Networks

## 1.1 What Is an Artificial Neural Network?

An **Artificial Neural Network (ANN)** is a machine-learning model made
of interconnected computational units called artificial neurons. These
neurons are organized into layers and use learnable parameters to
transform input data into predictions.

A typical fully connected ANN has:

1.  **Input layer:** Receives the input features.
2.  **Hidden layer(s):** Transform the input into intermediate
    representations.
3.  **Output layer:** Produces the final prediction.

For example, an ANN that predicts house prices might receive area,
number of rooms, and building age as input features and return an
estimated price.

## 1.2 The Basic Idea

An artificial neuron receives inputs, multiplies them by weights, adds a
bias, and applies an activation function.

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

$$
a=g(z)
$$

Here: - (x_i): input feature (i). - (w_i): weight associated with that
input. - (b): bias. - (z): pre-activation, or weighted sum plus bias. -
(g): activation function. - (a): neuron output, also called activation.

The ANN combines many such neurons to learn a mapping from inputs to
outputs.

## 1.3 How Does an ANN Learn?

A standard supervised ANN training process repeats these steps:

1.  **Forward propagation:** Calculate predictions.
2.  **Loss calculation:** Measure prediction error against the target.
3.  **Backpropagation:** Calculate gradients of the loss with respect to
    parameters.
4.  **Optimization:** Update weights and biases.
5.  **Repeat:** Continue over batches and epochs.

The central distinction is:

-   Forward propagation computes the output.
-   Backpropagation computes gradients.
-   The optimizer updates parameters.

These are related but different operations.

## 1.4 ANN as a Parameterized Function

A network can be written as:

$$
\hat{y}=f_{\theta}(x)
$$

where (`\theta`{=tex}) represents all trainable weights and biases.
Training seeks parameter values that minimize a chosen loss on the
training data while ideally generalizing to unseen data.

------------------------------------------------------------------------

# 2. What Problems Can ANN Solve?

  -----------------------------------------------------------------------
  Problem type            Example                 Typical output
  ----------------------- ----------------------- -----------------------
  Regression              Predicting house prices Continuous number

  Binary classification   Spam vs. not spam       One probability

  Multiclass              Classifying an animal   Class probabilities
  classification          into one of several     
                          classes                 

  Multilabel              Assigning multiple tags One probability per
  classification          to an image             label

  Forecasting             Predicting future       One or more future
                          demand                  values

  Representation learning Learning useful         Hidden feature vectors
                          features from raw       
                          inputs                  
  -----------------------------------------------------------------------

ANNs are flexible, but they are not automatically the best model for
every problem. On small tabular datasets, models such as
gradient-boosted trees can be highly competitive and may require less
tuning.

------------------------------------------------------------------------

# 3. Biological Neuron vs. Artificial Neuron

Biological neurons receive electrochemical signals through dendrites,
process them through complex biological mechanisms, and communicate
through axons and synapses. Artificial neurons are simplified
mathematical abstractions inspired loosely by this idea.

  -----------------------------------------------------------------------
  Biological concept                  Artificial-network analogy
  ----------------------------------- -----------------------------------
  Incoming signals                    Input values

  Synaptic strength                   Weight

  Integrated signal                   Weighted sum

  Firing behavior                     Activation function

  Neural connection changes           Parameter learning (only a loose
                                      analogy)
  -----------------------------------------------------------------------

The analogy is useful for intuition, but an artificial neuron is not a
realistic simulation of a biological neuron.

------------------------------------------------------------------------

# 4. Structure of an Artificial Neural Network

## 4.1 Input Layer

The input layer represents the features supplied to the model. If each
example has 5 features, the input vector can be written:

$$
x=
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4\\
x_5
\end{bmatrix}
\in\mathbb{R}^{5}
$$

Input layers generally do not have trainable weights of their own in a
basic dense ANN.

## 4.2 Hidden Layers

Hidden layers apply learned transformations to the input representation.
A dense layer calculates:

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

The activation of one layer becomes the input to the next.

## 4.3 Output Layer

The output layer is selected according to the task:

-   Regression: often linear output.
-   Binary classification: often one sigmoid output.
-   Mutually exclusive multiclass classification: often softmax outputs.
-   Multilabel classification: often one sigmoid output per label.

## 4.4 Architecture Notation

An architecture such as:

$$
4 \rightarrow 6 \rightarrow 3 \rightarrow 1
$$

means: - 4 input features. - 6 neurons in hidden layer 1. - 3 neurons in
hidden layer 2. - 1 output neuron.

The numbers represent units/features at each stage, not parameter
values.

------------------------------------------------------------------------

# 5. The Artificial Neuron: Mathematical Model

## 5.1 Weighted Sum

For (n) input features:

$$
s=\sum_{i=1}^{n}w_ix_i
$$

Each input is multiplied by a corresponding weight, and the products are
added.

## 5.2 Add the Bias

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

The bias shifts the neuron's response. It is a trainable parameter, just
like a weight.

## 5.3 Apply an Activation Function

$$
a=g(z)
$$

The full neuron formula is:

$$
\boxed{a=g\left(\sum_{i=1}^{n}w_ix_i+b\right)}
$$

## 5.4 Numerical Example

Suppose:

$$
x_1=2,\quad x_2=3
$$

$$
w_1=0.4,\quad w_2=-0.1,\quad b=0.2
$$

Then:

$$
z=(0.4)(2)+(-0.1)(3)+0.2
$$

$$
z=0.8-0.3+0.2=0.7
$$

Using ReLU:

$$
a=\max(0,0.7)=0.7
$$

Using sigmoid instead:

$$
a=\frac{1}{1+e^{-0.7}}\approx 0.6682
$$

The pre-activation is the same; the activation function determines the
output.

------------------------------------------------------------------------

# 6. Weights, Biases, Inputs, and Activations

## 6.1 Inputs

Inputs are the features presented to the model. Feature scaling can be
important because features with very different numeric ranges may make
optimization more difficult.

## 6.2 Weights

Weights control the contribution of input values to a neuron's
pre-activation. They are learned from data during training.

A weight's sign and magnitude should not be interpreted in isolation as
a complete measure of feature importance, particularly in a deep
network.

## 6.3 Biases

Biases allow the neuron to shift its pre-activation independently of the
weighted input sum:

$$
z=w^Tx+b
$$

Without bias, a linear neuron is constrained to have zero output when
its input vector is zero. A bias adds flexibility.

## 6.4 Pre-Activation vs. Activation

  Quantity   Meaning
  ---------- -------------------------------------------
  \(z\)      Weighted sum plus bias, before activation
  \(a\)      Output after activation

For a layer:

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

## 6.5 Parameters and Hyperparameters

**Parameters** are learned from data: - Weights. - Biases.

**Hyperparameters** are set by the model designer or training
procedure: - Number of layers and units. - Learning rate. - Batch
size. - Number of epochs. - Activation functions. - Regularization
strength.

------------------------------------------------------------------------

# 7. Activation Functions in ANN

Activation functions introduce nonlinearity into the network. Without
nonlinear activations, a stack of dense affine layers is equivalent to
one affine transformation and cannot represent the same rich family of
nonlinear mappings.

## 7.1 Linear Activation

$$
g(z)=z
$$

Useful for many regression output layers. If every layer is linear,
however, stacking layers does not create a nonlinear model.

## 7.2 Binary Step Function

$$
g(z)=
\begin{cases}
1, & z\geq 0\\
0, & z<0
\end{cases}
$$

It is associated with the classic perceptron. Its derivative is zero
almost everywhere and undefined at the threshold, so it is not normally
used for hidden layers trained by standard gradient descent.

## 7.3 Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Range:

$$
0<\sigma(z)<1
$$

Examples:

$$
\sigma(0)=0.5
$$

$$
\sigma(2)\approx 0.8808
$$

$$
\sigma(-2)\approx 0.1192
$$

Derivative:

$$
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}
$$

Uses: - Binary-classification output. - Gates in certain recurrent
architectures.

Limitation: sigmoid saturates at large positive or negative inputs,
where its derivative becomes small.

## 7.4 Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

Range:

$$
-1<\tanh(z)<1
$$

Derivative:

$$
\frac{d}{dz}\tanh(z)=1-\tanh^2(z)
$$

Tanh is zero-centered but can saturate for large absolute input values.

## 7.5 ReLU

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

Derivative away from zero:

$$
\operatorname{ReLU}'(z)=
\begin{cases}
1, & z>0\\
0, & z<0
\end{cases}
$$

At (z=0), the derivative is not classically defined; implementations
choose a convention, commonly zero.

Advantages: - Simple and efficient. - Does not saturate on the positive
side.

Limitation: units can become inactive if they repeatedly receive
negative pre-activations.

## 7.6 Leaky ReLU

$$
g(z)=
\begin{cases}
z, & z\geq 0\\
\alpha z, & z<0
\end{cases}
$$

Here (`\alpha`{=tex}) is a small positive slope. This preserves a
nonzero gradient on the negative side.

## 7.7 Softmax

For (K) logits:

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Properties: - Every output is positive. - Outputs sum to 1. - Commonly
used for mutually exclusive multiclass classification.

For numerical stability, implementations commonly subtract the maximum
logit before exponentiating:

$$
p_i=\frac{e^{z_i-c}}{\sum_{j=1}^{K}e^{z_j-c}},
\qquad c=\max_j z_j
$$

## 7.8 Activation Function Comparison

  ------------------------------------------------------------------------------------------
  Activation        Output range                         Common use        Main limitation
  ----------------- ------------------------------------ ----------------- -----------------
  Linear            ((-`\infty`{=tex},`\infty`{=tex}))   Regression output No nonlinearity

  Step              Usually ({0,1})                      Classic           Not suitable for
                                                         perceptron        standard gradient
                                                                           training

  Sigmoid           ((0,1))                              Binary output     Saturation

  Tanh              ((-1,1))                             Some              Saturation
                                                         hidden/gating     
                                                         uses              

  ReLU              (\[0,`\infty`{=tex}))                Hidden layers     Inactive units
                                                                           possible

  Leaky ReLU        ((-`\infty`{=tex},`\infty`{=tex}))   Hidden layers     Slope is a design
                                                                           choice

  Softmax           Positive values summing to 1         Multiclass output Can be
                                                                           overconfident;
                                                                           use stable
                                                                           implementation
  ------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 8. ANN Architecture and Layer Types

## 8.1 Fully Connected (Dense) Layer

Each neuron in a dense layer connects to every output from the previous
layer.

For a layer with (n\_{`\text{in}`{=tex}}) inputs and
(n\_{`\text{out}`{=tex}}) neurons:

$$
W\in\mathbb{R}^{n_{\text{out}}\times n_{\text{in}}}
$$

$$
b\in\mathbb{R}^{n_{\text{out}}}
$$

Under the column-vector convention, the parameter count is:

$$
\boxed{n_{\text{in}}n_{\text{out}}+n_{\text{out}}}
$$

## 8.2 Shallow vs. Deep ANN

-   **Shallow network:** Typically one hidden layer or a small number of
    layers.
-   **Deep network:** Multiple hidden layers.

The exact naming convention varies by context; what matters practically
is the network's architecture and capacity.

## 8.3 Feedforward Network

In a feedforward network, information moves from input toward output
without recurrent cycles in the computational graph.

A basic dense ANN is usually feedforward.

## 8.4 Residual Connections (Conceptual Extension)

Some deeper architectures add a skip connection:

$$
a^{[l+1]}=g(F(a^{[l]})+a^{[l]})
$$

This equation assumes the shapes match. If they do not, a projection may
be needed. Residual connections provide an additional path through which
information and gradients can flow.

------------------------------------------------------------------------

# 9. Forward Propagation

Forward propagation computes the network's output using the current
parameters.

## 9.1 Single Neuron

$$
z=w^Tx+b
$$

$$
a=g(z)
$$

## 9.2 One Hidden Layer

For a two-layer network:

$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$

$$
a^{[1]}=g^{[1]}(z^{[1]})
$$

$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$

$$
\hat{y}=a^{[2]}=g^{[2]}(z^{[2]})
$$

## 9.3 General Deep Network

Set:

$$
a^{[0]}=x
$$

For (l=1,`\ldots`{=tex},L):

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}=g^{[l]}(z^{[l]})
$$

The final prediction is:

$$
\hat{y}=a^{[L]}
$$

Forward propagation alone does not change the weights or biases.

## 9.4 Matrix Dimensions

Under the column-vector convention:

$$
a^{[l-1]}\in\mathbb{R}^{n_{l-1}}
$$

$$
W^{[l]}\in\mathbb{R}^{n_l\times n_{l-1}}
$$

$$
b^{[l]}\in\mathbb{R}^{n_l}
$$

Therefore:

$$
(n_l\times n_{l-1})(n_{l-1}\times 1)=(n_l\times 1)
$$

The inner dimensions must match.

## 9.5 Worked Forward-Pass Example

Architecture:

$$
2\rightarrow 2\rightarrow 1
$$

Input:

$$
x=\begin{bmatrix}1\\2\end{bmatrix}
$$

Parameters:

$$
W^{[1]}=
\begin{bmatrix}
0.1&0.2\\
0.3&0.4
\end{bmatrix},
\qquad
b^{[1]}=
\begin{bmatrix}0.1\\0.1\end{bmatrix}
$$

$$
W^{[2]}=\begin{bmatrix}0.5&-0.2\end{bmatrix},
\qquad b^{[2]}=0.1
$$

Hidden layer:

$$
z^{[1]}=W^{[1]}x+b^{[1]}
=
\begin{bmatrix}0.5\\1.1\end{bmatrix}
+
\begin{bmatrix}0.1\\0.1\end{bmatrix}
=
\begin{bmatrix}0.6\\1.2\end{bmatrix}
$$

With ReLU:

$$
a^{[1]}=\begin{bmatrix}0.6\\1.2\end{bmatrix}
$$

Output logit:

$$
z^{[2]}=(0.5)(0.6)+(-0.2)(1.2)+0.1=0.16
$$

Sigmoid output:

$$
\hat{y}=\sigma(0.16)\approx 0.5399
$$

This is the network's prediction for the given input and parameters.

------------------------------------------------------------------------

# 10. Loss Functions

A loss function measures the difference between the model prediction and
the target. It is calculated after the forward pass during supervised
training.

## 10.1 Mean Squared Error (MSE)

Commonly used for regression:

$$
\operatorname{MSE}=\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-\hat{y}^{(i)})^2
$$

For one example:

$$
\mathcal{L}=(y-\hat{y})^2
$$

Some formulations include a factor of (`\frac{1}{2}`{=tex}), which
simplifies derivatives:

$$
\mathcal{L}=\frac{1}{2}(\hat{y}-y)^2
$$

Both conventions are valid if used consistently.

## 10.2 Binary Cross-Entropy (BCE)

For binary target (y`\in`{=tex}{0,1}) and predicted probability
(`\hat{y}`{=tex}):

$$
\mathcal{L}=-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]
$$

For a batch:

$$
J=-\frac{1}{m}\sum_{i=1}^{m}
\left[
y^{(i)}\log\hat{y}^{(i)}
+(1-y^{(i)})\log(1-\hat{y}^{(i)})
\right]
$$

## 10.3 Categorical Cross-Entropy

For a one-hot target (y) and softmax output (p):

$$
\mathcal{L}=-\sum_{k=1}^{K}y_k\log p_k
$$

For a one-hot label, only the correct class contributes a nonzero target
term.

## 10.4 Logits vs. Probabilities

Some loss functions expect probabilities; others accept logits and apply
a stable transformation internally. Check the framework API carefully
and do not apply sigmoid or softmax twice.

------------------------------------------------------------------------

# 11. Backpropagation: The Core Idea

Backpropagation computes gradients of the loss with respect to the
network's parameters by applying the chain rule backward through the
computational graph.

It answers:

**How does a small change in a weight or bias affect the loss?**

If the loss is (J), we want derivatives such as:

$$
\frac{\partial J}{\partial W^{[l]}}
\qquad\text{and}\qquad
\frac{\partial J}{\partial b^{[l]}}
$$

These gradients are then used by an optimizer to update the parameters.

## 11.1 Forward vs. Backward Pass

  -----------------------------------------------------------------------
  Forward propagation                 Backpropagation
  ----------------------------------- -----------------------------------
  Moves from input toward output      Computes derivatives backward from
                                      the loss

  Calculates activations and          Calculates gradients
  predictions                         

  Uses current parameters             Uses derivatives through the
                                      computation graph

  Does not update parameters by       Supplies gradients to the optimizer
  itself                              
  -----------------------------------------------------------------------

## 11.2 Computational Graph View

For a simple neuron:

$$
x,w,b\rightarrow z=wx+b\rightarrow a=g(z)\rightarrow J
$$

Backpropagation starts from the loss and calculates how it depends on
(a), (z), (w), and (b).

------------------------------------------------------------------------

# 12. Derivatives and the Chain Rule

## 12.1 Basic Derivative Idea

A derivative measures how a function changes as its input changes.

If:

$$
y=f(x)
$$

then:

$$
\frac{dy}{dx}=f'(x)
$$

A derivative is local sensitivity, not necessarily a global description
of the function.

## 12.2 Chain Rule

If:

$$
y=f(u),\qquad u=g(x)
$$

then:

$$
\boxed{\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}}
$$

This rule is essential because neural networks are compositions of
functions.

## 12.3 Example

Let:

$$
u=2x,\qquad y=u^2
$$

Then:

$$
\frac{dy}{du}=2u,\qquad \frac{du}{dx}=2
$$

So:

$$
\frac{dy}{dx}=2u\cdot 2=4u=8x
$$

## 12.4 Derivative of MSE-Style Loss

For one example:

$$
J=\frac{1}{2}(\hat{y}-y)^2
$$

Then:

$$
\boxed{\frac{\partial J}{\partial\hat{y}}=\hat{y}-y}
$$

This derivative is the starting point for a simple regression
backpropagation example.

------------------------------------------------------------------------

# 13. Backpropagation in a Single Neuron

Consider one neuron:

$$
z=wx+b
$$

$$
\hat{y}=g(z)
$$

Use the loss:

$$
J=\frac{1}{2}(\hat{y}-y)^2
$$

We want (`\frac{\partial J}{\partial w}`{=tex}) and
(`\frac{\partial J}{\partial b}`{=tex}).

## Step 1: Derivative of Loss With Respect to Prediction

$$
\frac{\partial J}{\partial\hat{y}}=\hat{y}-y
$$

## Step 2: Derivative of Prediction With Respect to Pre-Activation

$$
\frac{\partial\hat{y}}{\partial z}=g'(z)
$$

## Step 3: Derivative of Pre-Activation With Respect to Weight

$$
\frac{\partial z}{\partial w}=x
$$

And:

$$
\frac{\partial z}{\partial b}=1
$$

## Step 4: Apply the Chain Rule

For the weight:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac{\partial J}{\partial\hat{y}}
\frac{\partial\hat{y}}{\partial z}
\frac{\partial z}{\partial w}
}
$$

Therefore:

$$
\boxed{\frac{\partial J}{\partial w}=(\hat{y}-y)g'(z)x}
$$

For the bias:

$$
\boxed{\frac{\partial J}{\partial b}=(\hat{y}-y)g'(z)}
$$

These equations show why the gradient depends on the error, the
activation derivative, and the input.

------------------------------------------------------------------------

# 14. Backpropagation in a Two-Layer Network

Consider a network with one hidden layer and one output layer:

$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$

$$
a^{[1]}=g^{[1]}(z^{[1]})
$$

$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$

$$
\hat{y}=g^{[2]}(z^{[2]})
$$

For a compact derivation, define the output error signal:

$$
\delta^{[2]}=\frac{\partial J}{\partial z^{[2]}}
$$

Then the output-layer gradients are:

$$
\boxed{\frac{\partial J}{\partial W^{[2]}}=\delta^{[2]}(a^{[1]})^T}
$$

$$
\boxed{\frac{\partial J}{\partial b^{[2]}}=\delta^{[2]}}
$$

The hidden-layer error signal is:

$$
\boxed{
\delta^{[1]}=
\left((W^{[2]})^T\delta^{[2]}\right)
\odot (g^{[1]})'(z^{[1]})
}
$$

Here (`\odot`{=tex}) denotes elementwise multiplication.

Then:

$$
\boxed{\frac{\partial J}{\partial W^{[1]}}=\delta^{[1]}x^T}
$$

$$
\boxed{\frac{\partial J}{\partial b^{[1]}}=\delta^{[1]}}
$$

These formulas use column vectors and a single training example. For a
batch, gradients are usually aggregated across examples according to the
chosen loss reduction.

## 14.1 Output Delta for Sigmoid + Binary Cross-Entropy

For a sigmoid output with binary cross-entropy, the derivative
simplifies to:

$$
\boxed{\delta^{[2]}=\hat{y}-y}
$$

This simplification assumes the usual sigmoid-plus-binary-cross-entropy
formulation and a single example (or the corresponding per-example
gradient).

## 14.2 Why Backpropagation Is Efficient

Rather than recomputing the influence of every parameter independently,
backpropagation reuses intermediate derivatives through the chain rule.
This makes gradient calculation practical for networks with many
parameters.

------------------------------------------------------------------------

# 15. Gradient Descent and Parameter Updates

Once gradients are calculated, an optimizer updates the parameters.

## 15.1 Gradient Descent

For a weight:

$$
\boxed{w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial J}{\partial w}}
$$

For a bias:

$$
\boxed{b_{\text{new}}=b_{\text{old}}-\eta\frac{\partial J}{\partial b}}
$$

Here (`\eta`{=tex}) is the learning rate.

The negative sign means that, for a sufficiently small step in ordinary
gradient descent, the update moves in the direction of decreasing loss
locally.

## 15.2 Numerical Example

Suppose:

$$
w_{\text{old}}=0.5,\qquad
\frac{\partial J}{\partial w}=0.8,\qquad
\eta=0.1
$$

Then:

$$
w_{\text{new}}=0.5-(0.1)(0.8)=0.42
$$

The weight changes from (0.5) to (0.42).

## 15.3 Learning Rate

-   Too large: updates may overshoot or diverge.
-   Too small: learning may be slow.
-   Appropriate: loss often decreases steadily, though training behavior
    depends on the model and data.

The best learning rate is not universal; it depends on the architecture,
optimizer, feature scales, and dataset.

------------------------------------------------------------------------

# 16. Complete Training Loop

A simplified training loop is:

1.  Select a batch of examples.
2.  Perform forward propagation.
3.  Calculate the batch loss.
4.  Compute gradients using backpropagation.
5.  Update weights and biases using the optimizer.
6.  Repeat for all batches.
7.  Repeat for the desired number of epochs.
8.  Evaluate on validation data.

Mathematically:

$$
\hat{y}=f_{\theta}(X)
$$

$$
J=\mathcal{L}(y,\hat{y})
$$

$$
\nabla_{\theta}J=\text{Backpropagation}(X,y,\theta)
$$

$$
\theta\leftarrow\text{OptimizerUpdate}(\theta,\nabla_{\theta}J)
$$

## 16.1 Epoch, Batch, and Iteration

-   **Epoch:** One pass through the entire training dataset.
-   **Batch:** A subset of training examples processed together.
-   **Iteration / step:** One parameter-update step, commonly after one
    batch.

If there are 1,000 training examples and batch size is 100, there are 10
batches per epoch when the data divides evenly and every batch is used.

------------------------------------------------------------------------

# 17. Weight Initialization

Weights should not generally all start at the same value. If all neurons
in a layer have identical weights and biases, they can receive identical
gradients and learn the same features, failing to break symmetry.

## 17.1 Zero Initialization

Setting every weight in a hidden layer to zero is generally a poor
choice for standard dense networks because identical neurons remain
symmetric. Biases may sometimes be initialized to zero, depending on the
architecture.

## 17.2 Xavier / Glorot Initialization

Often used with tanh-like activations. A common uniform form uses a
scale based on fan-in and fan-out:

$$
W_{ij}\sim U\left(
-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},
\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
\right)
$$

Exact implementation details vary by framework and initializer variant.

## 17.3 He Initialization

Often used with ReLU-like activations. A common normal form is:

$$
W_{ij}\sim\mathcal{N}\left(0,\frac{2}{n_{\text{in}}}\right)
$$

The expression gives the variance; the standard deviation is
(`\sqrt{2/n_{\text{in}}}`{=tex}).

## 17.4 Why Initialization Matters

Initialization influences the scale of activations and gradients at the
beginning of training. Poor scales can contribute to vanishing or
exploding signals and make optimization difficult.

------------------------------------------------------------------------

# 18. Learning Rate, Epochs, Batches, and Iterations

## 18.1 Learning Rate

Controls the step size of parameter updates. It is a hyperparameter for
basic SGD and is also used in adaptive optimizers.

## 18.2 Batch Size

The number of examples used to calculate a gradient estimate per update.
Smaller batches can create noisier updates; larger batches use more
memory and may change optimization behavior.

## 18.3 Epochs

The number of passes through the training dataset. More epochs do not
automatically mean better generalization; validation performance should
be monitored.

## 18.4 Training and Validation Data

-   **Training set:** Used to optimize parameters.
-   **Validation set:** Used to compare model settings and monitor
    generalization during development.
-   **Test set:** Reserved for a final, less biased evaluation after
    decisions are made.

Avoid repeatedly tuning against the test set because it gradually
becomes part of the model-selection process.

------------------------------------------------------------------------

# 19. Overfitting, Underfitting, and Generalization

## 19.1 Underfitting

A model underfits when it cannot capture the relevant patterns even in
the training data.

Possible causes: - Model too simple. - Insufficient training. - Features
lack useful information. - Excessive regularization. - Optimization
failure.

## 19.2 Overfitting

A model overfits when it learns training-specific details that do not
generalize well to unseen examples.

Typical pattern: - Training loss continues to decrease. - Validation
loss stops improving or begins increasing.

Possible causes: - Model too complex for the dataset. - Too little
training data. - Data leakage or unrepresentative splits. - Training for
too long without suitable regularization.

## 19.3 Generalization

Generalization is the ability to perform well on data not used to fit
the model. A low training loss alone does not guarantee good
generalization.

## 19.4 Learning Curves

Plot training and validation loss across epochs. The curves can help
distinguish optimization problems from overfitting, though they should
be interpreted with the dataset and task in mind.

------------------------------------------------------------------------

# 20. Regularization Techniques

## 20.1 L2 Regularization

L2 regularization penalizes large weights:

$$
J_{\text{total}}=J_{\text{data}}+\frac{\lambda}{2}\sum_j w_j^2
$$

Here (`\lambda`{=tex}) controls the strength of the penalty. Some
implementations use a slightly different scaling convention.

The penalty encourages smaller weights but does not guarantee that every
weight becomes small or zero.

## 20.2 L1 Regularization

$$
J_{\text{total}}=J_{\text{data}}+\lambda\sum_j|w_j|
$$

L1 regularization can encourage sparse weights. Its derivative is not
uniquely defined at zero, so optimization uses a subgradient or a
suitable algorithm.

## 20.3 Dropout

Dropout randomly sets selected activations to zero during training. It
is generally disabled during inference.

Dropout is not a substitute for sound validation, good data splits, or
appropriate architecture design.

## 20.4 Early Stopping

Stop training when validation performance has not improved for a chosen
patience period. Often, the best validation checkpoint is restored.

## 20.5 Data Augmentation

For data types where transformations preserve the label, augmentation
can increase the variety of training examples. Examples include image
flips or crops where semantically appropriate. Not every transformation
is valid for every task.

------------------------------------------------------------------------

# 21. Optimizers: SGD, Momentum, RMSprop, and Adam

## 21.1 Stochastic Gradient Descent (SGD)

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

Here (g_t) is the gradient estimate from the current batch.

## 21.2 Momentum

A common momentum formulation is:

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

Momentum accumulates a moving direction from recent gradients. Different
libraries may use slightly different conventions for the velocity and
learning-rate scaling.

## 21.3 RMSprop

RMSprop tracks a moving average of squared gradients:

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2
$$

Then a common update is:

$$
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
$$

Operations on vectors are elementwise.

## 21.4 Adam

Adam combines moving averages of gradients and squared gradients:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Bias-corrected estimates:

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

A common update is:

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

Adam is widely used because it adapts update scales per parameter, but
it is not always superior to SGD or other optimizers.

------------------------------------------------------------------------

# 22. ANN for Regression

Regression predicts one or more continuous values.

## 22.1 Architecture

For a single target:

$$
\text{Features}\rightarrow\text{Hidden layers}\rightarrow\text{One linear output}
$$

A common output equation is:

$$
\hat{y}=z^{[L]}
$$

A linear output does not constrain predictions to a particular interval.

## 22.2 Loss

Mean squared error:

$$
\operatorname{MSE}=\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-\hat{y}^{(i)})^2
$$

Mean absolute error:

$$
\operatorname{MAE}=\frac{1}{m}\sum_{i=1}^{m}|y^{(i)}-\hat{y}^{(i)}|
$$

MSE penalizes large errors quadratically; MAE grows linearly with the
absolute error.

## 22.3 Evaluation Metrics

Common metrics include: - MAE. - MSE. - RMSE. - (R\^2).

No single metric fully describes every regression task. Select metrics
based on the target units and business costs.

------------------------------------------------------------------------

# 23. ANN for Binary Classification

## 23.1 Output

A common architecture uses one sigmoid output:

$$
\hat{y}=\sigma(z^{[L]})
$$

The output is between 0 and 1 and is interpreted as the model's
estimated probability of the positive class.

## 23.2 Threshold

A common rule is:

$$
\hat{c}=
\begin{cases}
1, & \hat{y}\geq 0.5\\
0, & \hat{y}<0.5
\end{cases}
$$

A threshold of 0.5 is not mandatory. It can be adjusted to reflect the
cost of false positives and false negatives.

## 23.3 Evaluation Metrics

-   Accuracy.
-   Precision.
-   Recall.
-   F1-score.
-   ROC-AUC.
-   PR-AUC.
-   Confusion matrix.

Accuracy can be misleading when classes are imbalanced. For example, a
model that always predicts the majority class may have high accuracy but
poor minority-class recall.

------------------------------------------------------------------------

# 24. ANN for Multiclass Classification

## 24.1 Output Layer

For (K) mutually exclusive classes, the output layer often has (K)
logits followed by softmax:

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

## 24.2 Predicted Class

$$
\hat{c}=\arg\max_k p_k
$$

## 24.3 Loss

Categorical cross-entropy:

$$
\mathcal{L}=-\sum_{k=1}^{K}y_k\log p_k
$$

If labels are integer class IDs, a sparse categorical cross-entropy
implementation can avoid manually creating one-hot targets.

For multilabel tasks, use independent sigmoid outputs rather than
softmax when multiple labels may be true simultaneously.

------------------------------------------------------------------------

# 25. NumPy ANN From Scratch

This section implements a small binary-classification ANN with one
hidden layer. It uses column vectors internally for each example and
shows batch training using vectorized operations.

## 25.1 Important Shapes

For (m) examples and (n) features, this implementation stores input data
as rows:

-   `X`: ((m,n))
-   `W1`: ((n,h))
-   `b1`: ((1,h))
-   `W2`: ((h,1))
-   `b2`: ((1,1))

This row-per-example convention differs from the column-vector notation
used in some mathematical sections.

## 25.2 Complete NumPy Implementation

``` python
import numpy as np

def sigmoid(z):
    # Stable enough for this educational example.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0.0, z)

def binary_cross_entropy(y, y_hat):
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return -np.mean(
        y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)
    )

class SimpleANN:
    def __init__(self, input_size, hidden_size, seed=42):
        rng = np.random.default_rng(seed)

        # He-style initialization for the ReLU hidden layer.
        self.W1 = rng.normal(
            0.0, np.sqrt(2.0 / input_size),
            size=(input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))

        # Small random output weights.
        self.W2 = rng.normal(
            0.0, np.sqrt(1.0 / hidden_size),
            size=(hidden_size, 1)
        )
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        Y_hat = sigmoid(Z2)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "Y_hat": Y_hat,
        }
        return Y_hat, cache

    def backward(self, y, cache):
        X = cache["X"]
        A1 = cache["A1"]
        Y_hat = cache["Y_hat"]

        m = X.shape[0]

        # For sigmoid + binary cross-entropy:
        dZ2 = (Y_hat - y) / m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, grads, learning_rate):
        dW1, db1, dW2, db2 = grads

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def fit(self, X, y, epochs=1000, learning_rate=0.05):
        history = []

        for epoch in range(epochs):
            y_hat, cache = self.forward(X)
            loss = binary_cross_entropy(y, y_hat)
            grads = self.backward(y, cache)
            self.update(grads, learning_rate)
            history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

        return history

    def predict_proba(self, X):
        probabilities, _ = self.forward(X)
        return probabilities

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

## 25.3 Train on a Small Example

This toy dataset is for learning the mechanics, not for demonstrating
real-world performance.

``` python
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=float)

y = np.array([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
], dtype=float)

model = SimpleANN(input_size=2, hidden_size=4)
history = model.fit(
    X, y,
    epochs=2000,
    learning_rate=0.05
)

print("Probabilities:")
print(model.predict_proba(X))

print("Predicted classes:")
print(model.predict(X))
```

The model may need different settings to fit the tiny dataset well.
Neural-network training is sensitive to initialization, learning rate,
architecture, and number of updates.

## 25.4 What Happens in Each Training Iteration?

1.  `forward()` calculates hidden activations and predictions.
2.  `binary_cross_entropy()` calculates the loss.
3.  `backward()` applies the chain rule to calculate gradients.
4.  `update()` changes weights and biases.
5.  The process repeats.

This example implements full-batch gradient descent. It uses all
examples for every parameter update rather than mini-batches.

------------------------------------------------------------------------

# 26. ANN Using TensorFlow/Keras

Keras provides layers, automatic differentiation, optimizers, and
training loops.

## 26.1 Binary Classification Model

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()
```

This model has two hidden layers and one sigmoid output.

## 26.2 Train the Model

``` python
# X_train: shape (number_of_examples, 2)
# y_train: shape (number_of_examples,) or compatible binary-label shape

# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=32,
#     callbacks=[
#         keras.callbacks.EarlyStopping(
#             monitor="val_loss",
#             patience=5,
#             restore_best_weights=True
#         )
#     ]
# )
```

The data must be prepared first. For a real workflow, split training,
validation, and test data carefully, and fit preprocessing
transformations on training data only.

## 26.3 Make Predictions

``` python
# probabilities = model.predict(X_test)
# predicted_classes = (probabilities >= 0.5).astype(int)
```

`model.predict()` performs inference. It does not update model
parameters.

## 26.4 Regression Example

``` python
regression_model = keras.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)  # Linear output by default
])

regression_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

## 26.5 Multiclass Example

``` python
num_features = 20
num_classes = 4

multiclass_model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

multiclass_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

For integer class labels, sparse categorical cross-entropy is suitable
when the output is a softmax distribution and the labels are valid class
IDs from `0` to `num_classes - 1`.

------------------------------------------------------------------------

# 27. Evaluating an ANN

## 27.1 Classification Metrics

### Accuracy

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

$$
\operatorname{Precision}=\frac{TP}{TP+FP}
$$

### Recall

$$
\operatorname{Recall}=\frac{TP}{TP+FN}
$$

### F1-score

$$
F_1=2\cdot\frac{\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}
$$

Here: - (TP): true positives. - (TN): true negatives. - (FP): false
positives. - (FN): false negatives.

If a denominator is zero, the metric needs a defined convention, often
handled by a library parameter.

## 27.2 Regression Metrics

### Mean Absolute Error

$$
\operatorname{MAE}=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

### Root Mean Squared Error

$$
\operatorname{RMSE}=
\sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}
$$

### Coefficient of Determination

$$
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
$$

(R\^2) compares squared prediction error with the variation around the
target mean; it can be negative on test data when the model performs
worse than the mean baseline.

## 27.3 Validation Discipline

-   Keep the test set separate until final evaluation.
-   Avoid data leakage.
-   Use preprocessing fitted only on training data.
-   Compare against simple baselines.
-   Inspect class-specific metrics and residuals, not just one aggregate
    score.

------------------------------------------------------------------------

# 28. Common ANN Problems and Debugging

## 28.1 Loss Does Not Decrease

Possible causes: - Learning rate too high or too low. - Incorrect labels
or target shape. - Wrong loss function for the output activation. -
Gradient implementation error. - Inputs have poor scales. - Weights are
initialized badly. - Parameters are not actually being updated.

## 28.2 Exploding Gradients

Gradients become extremely large and updates destabilize training.
Possible remedies depend on the model and can include: - Appropriate
initialization. - Lower learning rate. - Gradient clipping. -
Normalization or architectural changes.

## 28.3 Vanishing Gradients

Gradients become very small as they propagate through many layers.
Saturating activations and repeated multiplication by small derivatives
can contribute. ReLU-like activations, suitable initialization, residual
connections, and normalization may help depending on the architecture.

## 28.4 Overfitting

Possible remedies: - More representative data. - Data augmentation where
valid. - Early stopping. - Regularization. - Dropout. - Reducing model
capacity.

## 28.5 Shape Errors

Check shapes before matrix multiplication:

``` python
print("X:", X.shape)
print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)
```

With row-per-example NumPy code, a common first-layer pattern is:

``` python
Z1 = X @ W1 + b1
```

where `X.shape == (batch_size, input_features)` and
`W1.shape == (input_features, hidden_units)`.

## 28.6 Data Leakage

Leakage occurs when information unavailable at real prediction time
enters training or model selection. Examples include fitting a scaler on
the entire dataset before splitting or accidentally using future
information to predict the past.

## 28.7 Debugging Checklist

-   [ ] Are input values numeric and finite?
-   [ ] Are feature and target shapes correct?
-   [ ] Are categorical variables encoded appropriately?
-   [ ] Are missing values handled?
-   [ ] Is scaling fitted on training data only?
-   [ ] Does output shape match the task?
-   [ ] Are activation and loss compatible?
-   [ ] Does loss change during training?
-   [ ] Are validation metrics monitored?
-   [ ] Is the final test set kept separate?

------------------------------------------------------------------------

# 29. ANN vs. Traditional Machine Learning

  ------------------------------------------------------------------------
  Aspect                  ANN                     Traditional ML models
                                                  (examples:
                                                  linear/logistic
                                                  regression, trees)
  ----------------------- ----------------------- ------------------------
  Feature learning        Hidden layers can learn Often relies more on
                          representations         input features and
                                                  model-specific
                                                  splits/transformations

  Data needs              May benefit from large  Some models work very
                          datasets, depending on  well on smaller datasets
                          task                    

  Compute                 Can be expensive for    Often less expensive for
                          large networks          simple models

  Interpretability        Deep networks can be    Linear models and small
                          difficult to interpret  trees can be easier to
                                                  inspect

  Data types              Flexible; especially    Tree and linear methods
                          useful in unstructured  are strong for many
                          data with suitable      tabular tasks
                          architectures           

  Tuning                  Architecture and        Hyperparameters still
                          optimization choices    matter, but may be fewer
                          matter                  for some models
  ------------------------------------------------------------------------

ANN is a family of models, not a universal replacement for every other
machine-learning method.

------------------------------------------------------------------------

# 30. ANN vs. CNN vs. RNN vs. Transformer

These terms describe architectures or architecture families with
different inductive biases.

  -----------------------------------------------------------------------
  Model family            Typical strength        Example
  ----------------------- ----------------------- -----------------------
  Basic dense ANN / MLP   Fixed-length feature    Tabular prediction
                          vectors                 

  CNN                     Local spatial patterns  Image classification
                          and shared filters      

  RNN / LSTM / GRU        Sequential state        Time-series and
                          processing              sequence tasks

  Transformer             Attention-based         Language, vision,
                          interactions across     multimodal tasks
                          tokens or elements      
  -----------------------------------------------------------------------

A CNN, RNN, or Transformer is also a neural network. "ANN" is sometimes
used broadly for all artificial neural networks and sometimes informally
to mean a basic dense multilayer perceptron (MLP). Clarify the intended
meaning in context.

------------------------------------------------------------------------

# 31. Practical Applications

## 31.1 Finance

-   Credit-risk scoring.
-   Fraud detection.
-   Forecasting and anomaly detection.

## 31.2 Healthcare

-   Risk prediction.
-   Medical-image analysis when using appropriate architectures.
-   Decision-support models subject to validation and clinical
    safeguards.

## 31.3 Retail and Business

-   Demand forecasting.
-   Customer-churn prediction.
-   Recommendation components.
-   Price or sales prediction.

## 31.4 Computer Vision

Dense ANN layers can classify extracted features, while CNNs or Vision
Transformers are often used to learn from image pixels.

## 31.5 Natural Language Processing

ANNs are fundamental to modern NLP, but sequence models and Transformers
are usually more appropriate than a plain dense network for raw text
sequences.

------------------------------------------------------------------------

# 32. Interview Questions and Answers

## Q1. What is an ANN?

An ANN is a machine-learning model composed of interconnected
computational units arranged into layers, with trainable weights and
biases.

## Q2. What is an artificial neuron?

A neuron computes a weighted sum plus bias and applies an activation
function:

$$
a=g(w^Tx+b)
$$

## Q3. Why do we need weights?

Weights control how strongly each input contributes to the
pre-activation.

## Q4. Why do we need bias?

Bias provides an adjustable offset independent of the weighted input
sum.

## Q5. What is an activation function?

A function applied to the pre-activation to produce the neuron's output.
Nonlinear activations enable networks to learn nonlinear mappings.

## Q6. What is forward propagation?

The computation that passes input through layers to produce a
prediction.

## Q7. What is backpropagation?

The chain-rule-based algorithm used to compute gradients of the loss
with respect to parameters.

## Q8. Does backpropagation update weights directly?

Backpropagation computes gradients. The optimizer uses those gradients
to update the parameters.

## Q9. What is a loss function?

A scalar objective that quantifies prediction error or another training
objective.

## Q10. What is gradient descent?

An optimization method that updates parameters in the negative gradient
direction, scaled by a learning rate.

## Q11. Why is ReLU commonly used?

It is computationally simple and avoids saturation on the positive side,
though inactive units can occur.

## Q12. Why is sigmoid used for binary classification?

It maps one logit into the interval ((0,1)), supporting a probability
interpretation for the positive class.

## Q13. Why is softmax used for multiclass classification?

It transforms logits into positive outputs that sum to 1 for mutually
exclusive classes.

## Q14. What is an epoch?

One pass through the full training dataset.

## Q15. What is a batch?

A subset of examples processed together for a training step.

## Q16. What is an iteration?

Usually, one optimizer update based on one batch.

## Q17. What is overfitting?

When a model fits training-specific patterns that do not generalize well
to unseen data.

## Q18. What is underfitting?

When a model fails to capture important patterns even in the training
data.

## Q19. How can overfitting be reduced?

Possible methods include early stopping, regularization, dropout, valid
data augmentation, more data, and reducing model capacity.

## Q20. What is the vanishing-gradient problem?

Gradients can become very small as they propagate backward through many
operations, slowing or preventing learning in earlier layers.

## Q21. Why not initialize every weight to zero?

Identical hidden neurons can remain symmetric and learn the same
features. Random or carefully designed initializations help break this
symmetry.

## Q22. What is the difference between a parameter and a hyperparameter?

Parameters are learned from data; hyperparameters control the model
design or training procedure.

## Q23. What is the role of the learning rate?

It controls the scale of parameter updates.

## Q24. Why split data into training, validation, and test sets?

Training fits parameters, validation supports model selection, and test
data provides a final evaluation on data not used for those decisions.

## Q25. Is ANN always better than decision trees or gradient boosting?

No. Performance depends on data size, feature types, task, architecture,
tuning, and evaluation protocol. Tree-based models can be excellent for
tabular data.

## Q26. What happens if every layer is linear?

The full network reduces to one affine transformation, so extra linear
layers do not add nonlinear representational power.

## Q27. What is the difference between classification and regression outputs?

Classification commonly uses probabilities or class scores; regression
commonly uses continuous numeric outputs.

## Q28. Why scale numerical features?

Scaling can improve numerical conditioning and make optimization easier,
especially when features have very different ranges.

## Q29. What is dropout?

A regularization method that randomly disables selected activations
during training and is typically disabled during inference.

## Q30. What is Adam?

An adaptive optimizer that uses moving averages of gradients and squared
gradients, with bias correction in its standard formulation.

------------------------------------------------------------------------

# 33. Practice Problems

Try these before reading the answer key.

## Problem 1: Neuron Output

Given:

$$
x_1=2,\quad x_2=4
$$

$$
w_1=0.5,\quad w_2=-0.25,\quad b=1
$$

Calculate (z) and the ReLU output.

## Problem 2: Sigmoid

Calculate:

$$
\sigma(0)
$$

## Problem 3: Layer Dimensions

A layer receives 5 input values and has 7 neurons. Under the
column-vector convention, what are the shapes of (W), (b), and (z)?

## Problem 4: Parameter Count

A dense layer has 10 inputs and 4 neurons. How many trainable parameters
are there when each neuron has one bias?

## Problem 5: Gradient Descent

Given:

$$
w=0.8,\quad \frac{\partial J}{\partial w}=-0.3,\quad \eta=0.1
$$

Calculate the updated weight.

## Problem 6: Activation

Why does a deep network need nonlinear activation functions if it is to
represent nonlinear mappings?

## Problem 7: Overfitting

Training loss continues to fall while validation loss rises. What might
this indicate, and name two possible responses.

## Problem 8: Binary Classification

A sigmoid output is (0.82). What is the predicted class under a
threshold of (0.5)?

## Problem 9: Multiclass Classification

What activation is commonly used for mutually exclusive multiclass
classification, and what property do its outputs have?

## Problem 10: Forward vs. Backpropagation

State the main job of each process.

------------------------------------------------------------------------

# 34. Answer Key

## Answer 1

$$
z=(0.5)(2)+(-0.25)(4)+1=1-1+1=1
$$

$$
\operatorname{ReLU}(1)=1
$$

## Answer 2

$$
\sigma(0)=\frac{1}{1+e^0}=0.5
$$

## Answer 3

$$
W\in\mathbb{R}^{7\times 5}
$$

$$
b\in\mathbb{R}^{7\times 1}
$$

$$
z\in\mathbb{R}^{7\times 1}
$$

## Answer 4

$$
10(4)+4=44
$$

There are 40 weights and 4 biases.

## Answer 5

$$
w_{\text{new}}=w-\eta\frac{\partial J}{\partial w}
$$

$$
w_{\text{new}}=0.8-(0.1)(-0.3)=0.83
$$

## Answer 6

Without nonlinear activations, compositions of affine layers collapse to
a single affine transformation. Nonlinear activations let the network
represent richer nonlinear functions.

## Answer 7

It may indicate overfitting. Possible responses include early stopping,
regularization, dropout, more representative training data, or reducing
model capacity.

## Answer 8

Since (0.82`\geq 0.5`{=tex}), the predicted class is 1.

## Answer 9

Softmax. Its outputs are positive and sum to 1, forming a distribution
across the mutually exclusive classes.

## Answer 10

-   Forward propagation computes activations and predictions.
-   Backpropagation computes gradients of the loss with respect to
    parameters.

The optimizer then uses those gradients to update parameters.

------------------------------------------------------------------------

# 35. Summary and Formula Sheet

## 35.1 Artificial Neuron

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

## 35.2 Layer Forward Pass

$$
\boxed{z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}}
$$

$$
\boxed{a^{[l]}=g^{[l]}(z^{[l]})}
$$

## 35.3 Sigmoid

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

$$
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}
$$

## 35.4 ReLU

$$
\boxed{\operatorname{ReLU}(z)=\max(0,z)}
$$

## 35.5 Softmax

$$
\boxed{p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}}
$$

## 35.6 Mean Squared Error

$$
\boxed{\operatorname{MSE}=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}
$$

## 35.7 Binary Cross-Entropy

$$
\boxed{
\mathcal{L}=-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]
}
$$

## 35.8 Gradient Descent

$$
\boxed{\theta_{\text{new}}=\theta_{\text{old}}-\eta\nabla_{\theta}J}
$$

## 35.9 Single-Neuron Gradients

For:

$$
z=wx+b,\quad \hat{y}=g(z),\quad J=\frac12(\hat{y}-y)^2
$$

the gradients are:

$$
\boxed{\frac{\partial J}{\partial w}=(\hat{y}-y)g'(z)x}
$$

$$
\boxed{\frac{\partial J}{\partial b}=(\hat{y}-y)g'(z)}
$$

## 35.10 Dense-Layer Parameter Count

$$
\boxed{\text{Parameters}=n_{\text{in}}n_{\text{out}}+n_{\text{out}}}
$$

## 35.11 Training Pipeline

$$
\boxed{
\text{Forward Pass}
\rightarrow
\text{Loss}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Optimizer Update}
}
$$

## 35.12 Final Takeaways

1.  An ANN consists of interconnected artificial neurons organized into
    layers.
2.  A neuron calculates a weighted sum, adds a bias, and applies an
    activation.
3.  Nonlinear activations allow a multilayer network to represent
    nonlinear mappings.
4.  Forward propagation computes predictions.
5.  A loss function measures the discrepancy between predictions and
    targets.
6.  Backpropagation calculates gradients using the chain rule.
7.  An optimizer uses gradients to update weights and biases.
8.  Initialization, learning rate, batch size, and regularization affect
    training.
9.  Validation data helps assess generalization and guide model
    selection.
10. Model evaluation should use metrics appropriate for the task and
    should avoid data leakage.
11. ANN is a broad family; CNNs, RNNs, and Transformers are
    neural-network architecture families with different structures and
    strengths.
12. NumPy implementations help build mathematical understanding, while
    frameworks such as TensorFlow/Keras provide automatic
    differentiation and optimized training.

------------------------------------------------------------------------

**Suggested study order:** First understand Sections 5--9 and calculate
the forward pass by hand. Then study the chain rule and backpropagation
in Sections 11--15. Finally, run the NumPy example and compare its
workflow with the Keras implementation.
