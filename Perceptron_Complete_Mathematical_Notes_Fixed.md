# Perceptron: Complete Mathematical and Conceptual Notes

## From Beginner to Advanced | Deep Learning | Mathematical Derivations | Python Implementation



---



## Table of Contents



1. Introduction to the Perceptron

2. Why Do We Need a Perceptron?

3. Biological Inspiration

4. What Is an Artificial Neuron?

5. Mathematical Components of a Perceptron

6. Mathematical Derivation of the Perceptron

7. Step Activation Function

8. Decision Boundary and Geometric Interpretation

9. Complete Numerical Example

10. Perceptron Learning Algorithm

11. Mathematical Derivation of the Learning Rule

12. Training a Perceptron Step by Step

13. Perceptron for AND, OR, and NOT Gates

14. Why a Single Perceptron Cannot Solve XOR

15. Perceptron Convergence Theorem

16. Perceptron With Multiple Inputs and Multiple Classes

17. Feature Scaling and Data Preparation

18. Perceptron From Scratch Using NumPy

19. Perceptron Using Scikit-learn

20. Advantages and Limitations

21. Perceptron vs. Logistic Regression vs. Multilayer Neural Network

22. Historical Significance

23. Common Interview Questions

24. Practice Problems

25. Summary and Key Formulas



---



# 1. Introduction to the Perceptron



## 1.1 What is a Perceptron?



A **Perceptron is a simple artificial neuron and one of the earliest mathematical models of a learning machine.** It takes one or more input features, multiplies them by corresponding weights, adds a bias, and passes the resulting value through an activation function to produce an output.



A standard perceptron is primarily used for binary classification.



Examples:



- Determining whether an email is spam or not spam.

- Predicting whether a student passes or fails.

- Classifying an image into one of two categories.

- Determining whether a transaction should be flagged for further investigation.



These are illustrative applications. A basic perceptron is generally too limited for complex real-world classification tasks.



## 1.2 The fundamental equation



The mathematical model is:



$$

z=\sum_{i=1}^{n}w_i x_i+b

$$



The output is:



$$

\hat y=f(z)

$$



For a traditional perceptron, the activation function is usually the step function:



$$

\hat y=

\begin{cases}

1 & \text{if }z\geq 0\\

0 & \text{if }z<0

\end{cases}

$$



Here:



- $x_i$ : input feature.

- $w_i$: weight associated with the input.

- $b$: bias.

- $z$: weighted sum before activation.

- $f(z)$: activation function.

- $\hat y$: predicted output.



The symbol $\hat y$ means the model's prediction, while $y$ generally represents the actual target.



## 1.3 The basic intuition



Imagine a teacher deciding whether a student passes an examination based on two features:



- $x_1$: marks obtained.

- $x_2$: attendance percentage.



The perceptron assigns weights to both features, calculates their combined score, and compares that score with a threshold.



If the score is sufficiently high, the student is classified as passing. Otherwise, the student is classified as failing.



The perceptron does not inherently understand students or examinations. It performs a mathematical calculation using the input values and learned parameters.



## 1.4 The most important idea



A perceptron learns a decision rule from examples by adjusting its weights and bias.



It does not usually learn the decision rule by explicitly programming every possible condition. Instead, it uses labelled training examples to improve its predictions.



---



# 2. Why Do We Need a Perceptron?



Consider a dataset:



| Study hours | Result |

|---:|---|

| 1 | Fail |

| 2 | Fail |

| 4 | Pass |

| 5 | Pass |



We want a model to learn the relationship between study hours and examination results.



We could manually write:



```python

if study_hours >= 3:

    result = "Pass"

else:

    result = "Fail"

```



However, this manually specifies the decision threshold.



A perceptron attempts to learn suitable parameters from training examples.



For one input feature:



$$

z=wx+b

$$



Suppose the learned parameters are:



$$

w=1,\qquad b=-3

$$



Then:



$$

z=x-3

$$



The prediction becomes:



$$

\hat y=

\begin{cases}

1 & x\geq 3\\

0 & x<3

\end{cases}

$$



The model has effectively learned a threshold of 3.



**Important:** This is an illustrative model. Actual training may produce different parameters, and the chosen labels and threshold depend on the dataset.



---



# 3. Biological Inspiration



## 3.1 Biological neuron



A biological neuron receives electrical and chemical signals through its dendrites. Its cell body integrates incoming signals, and its axon transmits signals to other cells when certain conditions are met.



A simplified conceptual representation is:



```text

Dendrites → Cell Body → Axon

```



## 3.2 Artificial neuron



An artificial neuron is a mathematical abstraction inspired by some aspects of biological neurons.



```text

Inputs → Weighted Sum → Activation Function → Output

```



The artificial neuron can be expressed as:



$$

z=w_1x_1+w_2x_2+\cdots+w_nx_n+b

$$



$$

\hat y=f(z)

$$



A rough conceptual correspondence is:



| Biological concept | Artificial counterpart |

|---|---|

| Incoming signals | Input features |

| Synaptic strengths | Weights |

| Integration of signals | Weighted sum |

| Simplified firing rule | Activation function |

| Signal sent onward | Output |



This is an analogy, not a literal biological equivalence. Artificial neurons are mathematical models and do not reproduce the full complexity of real neurons.



---



# 4. What Is an Artificial Neuron?



An artificial neuron processes information through a sequence of mathematical operations.



## 4.1 Inputs



Inputs are the features supplied to the model.



For example:



$$

x_1=4,\qquad x_2=7

$$



The two features could represent study hours and attendance, or any other pair of numerical measurements.



For $n$ features, we write:



$$

\mathbf{x}=

\begin{bmatrix}

x_1\\

x_2\\

\vdots\\

x_n

\end{bmatrix}

$$



The vector $\mathbf{x}$ contains all input features for one observation.



## 4.2 Weights



Each input has an associated weight.



$$

\mathbf{w}=

\begin{bmatrix}

w_1\\

w_2\\

\vdots\\

w_n

\end{bmatrix}

$$



Weights determine how strongly each feature contributes to the weighted sum.



For example:



$$

w_1=2,\qquad w_2=-1

$$



A positive weight increases the weighted sum when its input increases, assuming other values remain fixed.



A negative weight decreases the weighted sum when its input increases.



A weight of zero means that the corresponding feature contributes nothing directly to the weighted sum.



Weights should not automatically be interpreted as causal effects or as complete measures of feature importance.



## 4.3 Bias



Bias is an adjustable parameter that shifts the decision boundary.



The equation is:



$$

z=\sum_{i=1}^{n}w_ix_i+b

$$



Without bias:



$$

z=\sum_{i=1}^{n}w_ix_i

$$



The decision boundary would pass through the origin.



With bias, the boundary can shift away from the origin.



For example:



$$

z=2x-6

$$



The decision boundary is:



$$

2x-6=0

$$



Therefore:



$$

x=3

$$



Here, the bias allows the threshold to be positioned at $x=3$.



## 4.4 Weighted sum



The weighted sum combines all features with their corresponding weights and adds the bias.



For two inputs:



$$

z=w_1x_1+w_2x_2+b

$$



Suppose:



$$

x_1=3,\quad x_2=4

$$



$$

w_1=2,\quad w_2=0.5,\quad b=-1

$$



Then:



$$

z=(2)(3)+(0.5)(4)-1

$$



$$

z=6+2-1

$$



$$

\boxed{z=7}

$$



The value $z=7$ is called the pre-activation value.



## 4.5 Activation function



The activation function transforms the weighted sum into the output.



For a traditional binary perceptron:



$$

f(z)=

\begin{cases}

1 & z\geq 0\\

0 & z<0

\end{cases}

$$



Because $z=7\geq 0$:



$$

\boxed{\hat y=1}

$$



The complete operation is:



```text

Inputs

  ↓

Multiply by weights

  ↓

Add the weighted values

  ↓

Add bias

  ↓

Apply activation function

  ↓

Produce prediction

```



---



# 5. Mathematical Derivation of the Perceptron



## 5.1 Deriving the weighted sum



Suppose an observation has $n$ input features.



Each feature contributes according to its corresponding weight:



$$

w_1x_1,\;w_2x_2,\;\ldots,\;w_nx_n

$$



Adding these contributions gives:



$$

w_1x_1+w_2x_2+\cdots+w_nx_n

$$



Adding the bias gives:



$$

z=w_1x_1+w_2x_2+\cdots+w_nx_n+b

$$



Using summation notation:



$$

\boxed{z=\sum_{i=1}^{n}w_ix_i+b}

$$



This is the scalar form of the perceptron equation.



## 5.2 Deriving the vector form



Let:



$$

\mathbf{x}=

\begin{bmatrix}

x_1\\x_2\\\vdots\\x_n

\end{bmatrix},

\qquad

\mathbf{w}=

\begin{bmatrix}

w_1\\w_2\\\vdots\\w_n

\end{bmatrix}

$$



The dot product is:



$$

\mathbf{w}^{T}\mathbf{x}

=

\begin{bmatrix}

w_1&w_2&\cdots&w_n

\end{bmatrix}

\begin{bmatrix}

x_1\\x_2\\\vdots\\x_n

\end{bmatrix}

$$



Multiplying gives:



$$

\mathbf{w}^{T}\mathbf{x}

=w_1x_1+w_2x_2+\cdots+w_nx_n

$$



Therefore:



$$

\boxed{z=\mathbf{w}^{T}\mathbf{x}+b}

$$



The prediction is:



$$

\boxed{\hat y=f(\mathbf{w}^{T}\mathbf{x}+b)}

$$



The vector representation is useful for mathematical analysis and efficient implementation.



## 5.3 Augmented-vector representation



We can combine the bias and weights into a single vector.



Define:



$$

\tilde{\mathbf{x}}=

\begin{bmatrix}

x_1\\x_2\\\vdots\\x_n\\1

\end{bmatrix}

$$



and:



$$

\tilde{\mathbf{w}}=

\begin{bmatrix}

w_1\\w_2\\\vdots\\w_n\\b

\end{bmatrix}

$$



Then:



$$

\boxed{z=\tilde{\mathbf{w}}^{T}\tilde{\mathbf{x}}}

$$



Why does this work?



$$

\tilde{\mathbf{w}}^{T}\tilde{\mathbf{x}}

=w_1x_1+\cdots+w_nx_n+b(1)

$$



Therefore, the bias is simply the weight associated with an additional constant input equal to 1.



This representation is useful in theoretical derivations of the perceptron learning algorithm.



---



# 6. Step Activation Function



## 6.1 Definition



The step activation function is defined as:



$$

f(z)=

\begin{cases}

1 & z\geq 0\\

0 & z<0

\end{cases}

$$



It converts a continuous weighted sum into a binary output.



For example:



| $z$ | Output |

|---:|---:|

| -10 | 0 |

| -2 | 0 |

| -0.1 | 0 |

| 0 | 1 |

| 0.5 | 1 |

| 3 | 1 |

| 10 | 1 |



The convention at zero can vary. Here, zero is assigned to class 1.



## 6.2 Why use a step function?



The step function is useful when the task requires a direct binary decision.



For example:



- Class 0: Not approved.

- Class 1: Approved.



The function produces only two possible outputs.



However, the output does not represent a probability.



If the perceptron outputs 1, it does not mean there is a 100% probability that the positive class is correct.



## 6.3 Mathematical discontinuity



The step function is discontinuous at $z=0$.



For values just below zero, the output is 0. At zero, the output becomes 1.



Consequently, the ordinary derivative is zero wherever the function is locally constant and undefined at the discontinuity.



This matters because gradient-based optimization usually relies on derivatives to determine how parameters should change.



The classic perceptron therefore uses a mistake-driven update rule rather than ordinary backpropagation through its step activation.



---



# 7. Decision Boundary and Geometric Interpretation



The decision boundary is one of the most important concepts in perceptron theory.



## 7.1 What is a decision boundary?



A decision boundary is the set of input points where the model changes its predicted class.



For a perceptron, the boundary occurs where:



$$

z=0

$$



Therefore:



$$

\boxed{\mathbf{w}^{T}\mathbf{x}+b=0}

$$



For two input features:



$$

\boxed{w_1x_1+w_2x_2+b=0}

$$



This equation represents a straight line in two-dimensional feature space, provided the weights are not both zero.



## 7.2 Deriving the line equation



Start with:



$$

w_1x_1+w_2x_2+b=0

$$



Assume $w_2\neq 0$.



Rearrange:



$$

w_2x_2=-w_1x_1-b

$$



Divide by $w_2$:



$$

x_2=-\frac{w_1}{w_2}x_1-\frac{b}{w_2}

$$



Compare this with the familiar straight-line equation:



$$

y=mx+c

$$



We identify:



$$

\boxed{m=-\frac{w_1}{w_2}}

$$



and:



$$

\boxed{c=-\frac{b}{w_2}}

$$



Thus, the ratio of the weights determines the slope, while the bias affects the intercept.



If $w_2=0$ and $w_1\neq 0$, the boundary is the vertical line:



$$

x_1=-\frac{b}{w_1}

$$



## 7.3 Numerical example



Suppose:



$$

w_1=2,\qquad w_2=3,\qquad b=-6

$$



The decision boundary is:



$$

2x_1+3x_2-6=0

$$



Solving for $x_2$:



$$

3x_2=6-2x_1

$$



$$

\boxed{x_2=2-\frac{2}{3}x_1}

$$



This is a straight line.



The model predicts class 1 when:



$$

2x_1+3x_2-6\geq 0

$$



and class 0 when:



$$

2x_1+3x_2-6<0

$$



## 7.4 The role of the weight vector



The weight vector is:



$$

\mathbf{w}=

\begin{bmatrix}

w_1\\w_2

\end{bmatrix}

$$



It is perpendicular to the decision boundary.



For the example above:



$$

\mathbf{w}=

\begin{bmatrix}

2\\3

\end{bmatrix}

$$



Any movement perpendicular to the boundary changes the weighted sum most directly.



The weights therefore determine the orientation of the boundary, while the bias determines its position relative to the origin.



## 7.5 Higher dimensions



The same equation applies with any number of features:



$$

\mathbf{w}^{T}\mathbf{x}+b=0

$$



- One feature: a point on the one-dimensional axis.

- Two features: a line.

- Three features: a plane.

- More features: a hyperplane.



A hyperplane is the higher-dimensional generalization of a line or plane.



This leads to a central limitation of the basic perceptron: its decision boundary is linear in the input features.



---



# 8. Complete Numerical Example



Consider a perceptron with two inputs.



Given:



$$

x_1=3,\qquad x_2=2

$$



Weights:



$$

w_1=2,\qquad w_2=-1

$$



Bias:



$$

b=-3

$$



The model is:



$$

z=w_1x_1+w_2x_2+b

$$



## Step 1: Multiply each input by its weight



$$

w_1x_1=(2)(3)=6

$$



$$

w_2x_2=(-1)(2)=-2

$$



## Step 2: Add the weighted inputs



$$

6+(-2)=4

$$



## Step 3: Add the bias



$$

z=4-3

$$



$$

\boxed{z=1}

$$



## Step 4: Apply the step function



Since:



$$

z=1\geq 0

$$



the prediction is:



$$

\boxed{\hat y=1}

$$



## Step 5: Change the input



Now suppose:



$$

x_1=1,\qquad x_2=2

$$



Then:



$$

z=(2)(1)+(-1)(2)-3

$$



$$

z=2-2-3=-3

$$



Since $z<0$:



$$

\boxed{\hat y=0}

$$



The same weights and bias can classify different inputs into different classes.



---



# 9. Perceptron Learning Algorithm



## 9.1 What does learning mean?



Initially, the perceptron may have arbitrary weights and bias.



Its predictions may be incorrect.



Training adjusts the parameters when the model makes classification mistakes.



The objective is to find parameters that correctly classify the training examples, when such a solution exists within the model's hypothesis class.



## 9.2 The perceptron update rule



For labels encoded as 0 and 1, define the error:



$$

e=y-\hat y

$$



The update rules are:



$$

\boxed{w_i\leftarrow w_i+\eta(y-\hat y)x_i}

$$



and:



$$

\boxed{b\leftarrow b+\eta(y-\hat y)}

$$



where:



- $y$: actual label.

- $\hat y$: predicted label.

- $x_i$: input feature.

- $\eta$: learning rate.

- $w_i$: weight.

- $b$: bias.



The arrow $\leftarrow$ means that the parameter is replaced by its updated value.



## 9.3 Why is the learning rate needed?



The learning rate controls the magnitude of the update.



For example:



$$

\eta=1

$$



produces a full perceptron update.



If:



$$

\eta=0.1

$$



the update is ten times smaller.



In the standard perceptron convergence setting, a positive constant learning rate does not change whether a linearly separable dataset is separable by the algorithm; it scales the parameter trajectory. In practical implementations, however, the learning rate can affect numerical behavior and stopping behavior.



## 9.4 The three possible cases



### Case 1: Correctly predicted class 1



Suppose:



$$

y=1,\qquad \hat y=1

$$



Then:



$$

e=1-1=0

$$



Therefore:



$$

\Delta w_i=0

$$



$$

\Delta b=0

$$



No update is required.



### Case 2: Correctly predicted class 0



Suppose:



$$

y=0,\qquad \hat y=0

$$



Then:



$$

e=0-0=0

$$



Again, no update is required.



### Case 3: Incorrect prediction



Suppose:



$$

y=1,\qquad \hat y=0

$$



Then:



$$

e=1-0=1

$$



The updates are:



$$

w_i\leftarrow w_i+\eta x_i

$$



$$

b\leftarrow b+\eta

$$



The model increases the score for that input.



Now suppose:



$$

y=0,\qquad \hat y=1

$$



Then:



$$

e=0-1=-1

$$



The updates are:



$$

w_i\leftarrow w_i-\eta x_i

$$



$$

b\leftarrow b-\eta

$$



The model decreases the score for that input.



This is the core idea: **move the decision function in a direction that corrects the current mistake.**



---



# 10. Mathematical Derivation of the Learning Rule



## 10.1 Start with the prediction error



For labels in $\{0,1\}$:



$$

e=y-\hat y

$$



We want to adjust the parameters when $e\neq 0$.



The classic perceptron rule is:



$$

\Delta w_i=\eta e x_i

$$



$$

\Delta b=\eta e

$$



Therefore:



$$

\boxed{w_i^{\text{new}}=w_i^{\text{old}}+\eta e x_i}

$$



$$

\boxed{b^{\text{new}}=b^{\text{old}}+\eta e}

$$



## 10.2 Why multiply the error by the input?



The weight $w_i$ controls how much feature $x_i$ contributes to the weighted sum.



If $x_i$ is large, changing its weight has a larger effect on the score for that example.



If $x_i=0$, changing its weight does not affect the score for that particular example.



The update therefore scales the change in each weight by its corresponding input.



This explains the structure of the rule, although it is not a gradient-descent derivation of the discontinuous step-function model.



## 10.3 Why update the bias separately?



The bias affects the score regardless of the feature values:



$$

z=\sum_i w_ix_i+b

$$



Its update is:



$$

\Delta b=\eta e

$$



The bias allows the model to shift its boundary even when the input values are zero.



Without a bias update, the decision boundary could remain unnecessarily constrained.



## 10.4 Derivation using an augmented vector



Recall:



$$

\tilde{\mathbf{x}}=

\begin{bmatrix}

x_1\\x_2\\\vdots\\x_n\\1

\end{bmatrix}

$$



and:



$$

\tilde{\mathbf{w}}=

\begin{bmatrix}

w_1\\w_2\\\vdots\\w_n\\b

\end{bmatrix}

$$



The entire update can be written as:



$$

\boxed{

\tilde{\mathbf{w}}_{\text{new}}

=

\tilde{\mathbf{w}}_{\text{old}}

+\eta(y-\hat y)\tilde{\mathbf{x}}

}

$$



Expanding the final coordinate produces the bias update because the last input is 1.



## 10.5 A subtle but important distinction



The perceptron learning rule is not the same as ordinary gradient descent on binary cross-entropy.



The classic perceptron:



- Uses a hard classification decision.

- Updates parameters on misclassified examples.

- Does not require a differentiable probability output.

- Does not directly estimate calibrated class probabilities.



Logistic regression, by contrast, typically uses a sigmoid function and an explicitly defined probabilistic loss.



---



# 11. Training a Perceptron Step by Step



Consider a one-dimensional training example:



$$

x=2,\qquad y=1

$$



Initial parameters:



$$

w=0,\qquad b=0

$$



Learning rate:



$$

\eta=1

$$



## Step 1: Calculate the score



$$

z=wx+b

$$



$$

z=(0)(2)+0=0

$$



## Step 2: Make a prediction



Using the convention that $z\geq0$ gives class 1:



$$

\hat y=1

$$



## Step 3: Calculate the error



$$

e=y-\hat y=1-1=0

$$



## Step 4: Update the parameters



$$

w_{\text{new}}=0+(1)(0)(2)=0

$$



$$

b_{\text{new}}=0+(1)(0)=0

$$



No update is required because the example was already classified correctly.



Now consider a different initial bias:



$$

w=0,\qquad b=-1

$$



The score becomes:



$$

z=(0)(2)-1=-1

$$



Therefore:



$$

\hat y=0

$$



The error is:



$$

e=1-0=1

$$



Update the weight:



$$

w_{\text{new}}=0+(1)(1)(2)=2

$$



Update the bias:



$$

b_{\text{new}}=-1+(1)(1)=0

$$



The new parameters are:



$$

\boxed{w=2,\qquad b=0}

$$



For the same input $x=2$:



$$

z=(2)(2)+0=4

$$



The prediction is now:



$$

\hat y=1

$$



The mistake has been corrected.



Note that one update corrects this particular example. It does not guarantee that every other training example is also classified correctly.



---



# 12. Perceptron for Boolean Logic Gates



Logic gates are useful because they demonstrate exactly what a single perceptron can and cannot represent.



Use binary inputs:



$$

x_1,x_2\in\{0,1\}

$$



The perceptron output is:



$$

\hat y=

\begin{cases}

1 & w_1x_1+w_2x_2+b\geq0\\

0 & \text{otherwise}

\end{cases}

$$



## 12.1 AND gate



The AND gate outputs 1 only when both inputs are 1.



| $x_1$ | $x_2$ | AND output |

|---:|---:|---:|

| 0 | 0 | 0 |

| 0 | 1 | 0 |

| 1 | 0 | 0 |

| 1 | 1 | 1 |



Choose:



$$

w_1=1,\quad w_2=1,\quad b=-1.5

$$



Then:



$$

z=x_1+x_2-1.5

$$



Evaluate each input.



| Inputs | $z$ | Prediction |

|---|---:|---:|

| (0, 0) | -1.5 | 0 |

| (0, 1) | -0.5 | 0 |

| (1, 0) | -0.5 | 0 |

| (1, 1) | 0.5 | 1 |



Therefore, a single perceptron can represent the AND gate.



## 12.2 OR gate



The OR gate outputs 1 when at least one input is 1.



| $x_1$ | $x_2$ | OR output |

|---:|---:|---:|

| 0 | 0 | 0 |

| 0 | 1 | 1 |

| 1 | 0 | 1 |

| 1 | 1 | 1 |



Choose:



$$

w_1=1,\quad w_2=1,\quad b=-0.5

$$



Then:



$$

z=x_1+x_2-0.5

$$



| Inputs | $z$ | Prediction |

|---|---:|---:|

| (0, 0) | -0.5 | 0 |

| (0, 1) | 0.5 | 1 |

| (1, 0) | 0.5 | 1 |

| (1, 1) | 1.5 | 1 |



Therefore, a single perceptron can represent the OR gate.



## 12.3 NOT gate



The NOT gate has one input.



| $x$ | NOT output |

|---:|---:|

| 0 | 1 |

| 1 | 0 |



Choose:



$$

w=-1,\qquad b=0.5

$$



Then:



$$

z=-x+0.5

$$



For $x=0$:



$$

z=0.5\Rightarrow\hat y=1

$$



For $x=1$:



$$

z=-0.5\Rightarrow\hat y=0

$$



Thus, the NOT gate is also representable by a single perceptron.



## 12.4 What do these examples prove?



AND, OR, and NOT can all be represented using a single perceptron.



However, this does not mean that a single perceptron can represent every Boolean function.



The XOR gate provides a counterexample.



---



# 13. Why a Single Perceptron Cannot Solve XOR



## 13.1 XOR truth table



The XOR operation outputs 1 when the two inputs differ.



| $x_1$ | $x_2$ | XOR output |

|---:|---:|---:|

| 0 | 0 | 0 |

| 0 | 1 | 1 |

| 1 | 0 | 1 |

| 1 | 1 | 0 |



The positive examples are:



$$

(0,1),\quad(1,0)

$$



The negative examples are:



$$

(0,0),\quad(1,1)

$$



A single perceptron would need one straight-line decision boundary that separates the two positive examples from the two negative examples.



No such line exists.



## 13.2 Mathematical proof of impossibility



Suppose the perceptron has parameters $w_1,w_2,b$.



Its score is:



$$

z=w_1x_1+w_2x_2+b

$$



The model predicts class 1 when:



$$

z\geq0

$$



To correctly classify XOR, the point $(0,0)$ must be classified as 0:



$$

b<0

$$



The point $(1,1)$ must also be classified as 0:



$$

w_1+w_2+b<0

$$



The point $(1,0)$ must be classified as 1:



$$

w_1+b\geq0

$$



The point $(0,1)$ must be classified as 1:



$$

w_2+b\geq0

$$



Adding the last two inequalities gives:



$$

w_1+w_2+2b\geq0

$$



But the requirement for $(1,1)$ gives:



$$

w_1+w_2+b<0

$$



Combining these inequalities would require:



$$

-b\leq w_1+w_2<-b

$$



which is impossible.



Therefore:



$$

\boxed{\text{A single linear perceptron cannot represent XOR.}}

$$



This is a mathematical limitation, not merely a problem with the choice of learning rate or training duration.



## 13.3 How can a neural network solve XOR?



A multilayer neural network can create a nonlinear decision boundary by combining multiple neurons.



For example, a network with a hidden layer can compute intermediate features and then combine them.



Using the Boolean identity:



$$

\boxed{XOR=(A\ OR\ B)\ AND\ NOT(A\ AND\ B)}

$$



a network can implement XOR using hidden units that compute OR and AND, followed by an output unit that combines their results.



More generally, multilayer networks can learn useful intermediate representations that make some problems separable in the transformed feature space.



**Important:** Adding hidden layers is not a guarantee that every training process will succeed. Architecture, data, optimization, and parameter initialization also matter.



---



# 14. Perceptron Convergence Theorem



## 14.1 What is convergence?



In machine learning, convergence means that the learning procedure reaches a state satisfying its specified stopping condition.



For a perceptron, the classic result is that it will make only finitely many mistakes when trained on a finite, linearly separable dataset under the standard assumptions.



It does not guarantee convergence on every dataset.



## 14.2 Linear separability



A binary dataset is linearly separable if there exists a vector $\mathbf{w}^{*}$ and bias $b^{*}$ such that all positive examples fall on one side of the decision boundary and all negative examples fall on the other.



Using labels:



$$

y_i\in\{-1,+1\}

$$



the condition is:



$$

\boxed{y_i(\mathbf{w}^{*T}\mathbf{x}_i+b^{*})>0}

$$



for every training example $i$.



The use of labels $-1$ and $+1$ simplifies the mathematical expression.



## 14.3 The perceptron update with signed labels



With signed labels, the prediction score is:



$$

s_i=\mathbf{w}^{T}\mathbf{x}_i+b

$$



A mistake occurs when the sign of $s_i$ disagrees with $y_i$.



The update is:



$$

\boxed{\mathbf{w}\leftarrow\mathbf{w}+\eta y_i\mathbf{x}_i}

$$



$$

\boxed{b\leftarrow b+\eta y_i}

$$



for a misclassified example.



This is equivalent in spirit to the 0/1-label update, but the exact algebra uses a different label encoding.



## 14.4 Intuition behind the theorem



If a dataset is linearly separable, a suitable boundary exists.



When the perceptron misclassifies an example, the update moves its parameters in a direction that improves the signed score for that example.



Under the theorem's assumptions, the updates cannot continue making mistakes forever.



The theorem is especially useful because it establishes a finite-mistake guarantee without requiring the perceptron to optimize a smooth differentiable loss.



## 14.5 A more mathematical version



Assume:



1. The dataset contains finitely many training examples.

2. The examples are linearly separable.

3. The input vectors have bounded norm.

4. There is a strictly positive separation margin.



Using augmented vectors to include the bias, suppose:



$$

\|\tilde{\mathbf{x}}_i\|\leq R

$$



and there is a unit vector $\mathbf{u}$ such that:



$$

y_i\mathbf{u}^{T}\tilde{\mathbf{x}}_i\geq\gamma>0

$$



for every example.



Here:



- $R$ is an upper bound on the augmented input norm.

- $\gamma$ is a positive separation margin.

- $\mathbf{u}$ is a unit-length separating parameter vector.



With the standard mistake-driven update:



$$

\tilde{\mathbf{w}}_{t+1}

=

\tilde{\mathbf{w}}_t+\eta y_i\tilde{\mathbf{x}}_i

$$



the classic perceptron mistake bound is:



$$

\boxed{M\leq\left(\frac{R}{\gamma}\right)^2}

$$



where $M$ is the number of mistakes, under the stated assumptions.



The bound is independent of the number of possible parameter vectors and does not imply that the algorithm finds the maximum-margin separator.



A larger margin makes the theoretical bound smaller, while larger input norms make the bound larger.



---



# 15. Perceptron With Multiple Inputs and Multiple Classes



## 15.1 Multiple input features



A perceptron is not restricted to two inputs.



For $n$ input features:



$$

z=w_1x_1+w_2x_2+\cdots+w_nx_n+b

$$



For example, a student classification model might use:



$$

\mathbf{x}=

\begin{bmatrix}

\text{study hours}\\

\text{attendance}\\

\text{assignment score}\\

\text{previous exam score}

\end{bmatrix}

$$



The model learns one weight for each feature and a bias.



The weighted sum is:



$$

z=\mathbf{w}^{T}\mathbf{x}+b

$$



The decision boundary is a hyperplane in the corresponding feature space.



## 15.2 Multiclass classification



The basic perceptron is a binary classifier.



For multiple classes, one common extension is a multiclass perceptron with one score per class.



Suppose there are $K$ classes.



Each class $k$ has its own weight vector and bias:



$$

s_k=\mathbf{w}_k^{T}\mathbf{x}+b_k

$$



The predicted class is:



$$

\boxed{\hat y=\operatorname{arg\,max}_{k\in\{1,\ldots,K\}}s_k}

$$



The model chooses the class with the highest score.



This is a multiclass extension, not the same as a single binary perceptron.



## 15.3 One-vs-rest classification



Another approach is to train one binary classifier for each class.



For example, in a three-class problem:



- Classifier 1: Class A vs. not A.

- Classifier 2: Class B vs. not B.

- Classifier 3: Class C vs. not C.



A decision strategy is then needed to select a class from the individual outputs or scores.



This is called one-vs-rest classification.



---



# 16. Feature Scaling and Data Preparation



Although a perceptron is mathematically simple, the quality of its input representation matters.



## 16.1 Why feature scales matter



Suppose the input features are:



- Age: between 18 and 70.

- Annual income: between 100,000 and 5,000,000.



The income feature has a much larger numerical scale.



Consequently, changes in its weight may have a much larger effect on the weighted sum than changes in the age weight.



Feature scaling can improve numerical behavior and make learning rates easier to choose.



## 16.2 Standardization



A common standardization formula is:



$$

\boxed{x'=\frac{x-\mu}{\sigma}}

$$



where:



- $x$: original feature value.

- $\mu$: training-set mean.

- $\sigma$: training-set standard deviation.

- $x'$: standardized feature value.



The transformed feature has mean approximately zero and standard deviation approximately one when evaluated on the same training data, provided the standard deviation is nonzero.



Calculate the scaling parameters using the training data only. Reuse those parameters to transform validation and test data.



This prevents information leakage from the evaluation data.



## 16.3 Missing values and categorical features



A basic numerical perceptron cannot directly process arbitrary missing values or text categories.



Possible preparation steps include:



- Imputing missing numerical values.

- Encoding categorical features.

- Scaling numerical features.

- Checking for invalid values.

- Separating training and test datasets.



The exact preprocessing pipeline depends on the dataset.



## 16.4 Label encoding



The standard 0/1 perceptron uses:



$$

y\in\{0,1\}

$$



The signed-label formulation uses:



$$

y\in\{-1,+1\}

$$



Do not mix the two formulations without adjusting the prediction and update rules accordingly.



---



# 17. Perceptron From Scratch Using NumPy



The following implementation demonstrates the classic perceptron algorithm for binary classification.



It uses:



- A step activation function.

- A learning rate.

- A weight vector.

- A bias.

- Mistake-driven updates.



```python

import numpy as np





class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=100):

        self.learning_rate = learning_rate

        self.epochs = epochs

        self.weights = None

        self.bias = 0.0



    def activation(self, z):

        return 1 if z >= 0 else 0



    def predict_one(self, x):

        z = np.dot(x, self.weights) + self.bias

        return self.activation(z)



    def fit(self, X, y):

        n_samples, n_features = X.shape



        self.weights = np.zeros(n_features)

        self.bias = 0.0



        for epoch in range(self.epochs):

            mistakes = 0



            for xi, target in zip(X, y):

                prediction = self.predict_one(xi)



                error = target - prediction



                if error != 0:

                    self.weights += (

                        self.learning_rate * error * xi

                    )



                    self.bias += (

                        self.learning_rate * error

                    )



                    mistakes += 1



            print(

                f"Epoch {epoch + 1}: "

                f"Mistakes = {mistakes}"

            )



            if mistakes == 0:

                print("Training converged.")

                break



        return self



    def predict(self, X):

        return np.array([

            self.predict_one(xi) for xi in X

        ])

```



## 17.1 Train the perceptron on AND



```python

import numpy as np



X = np.array([

    [0, 0],

    [0, 1],

    [1, 0],

    [1, 1]

])



y = np.array([0, 0, 0, 1])



model = Perceptron(

    learning_rate=0.1,

    epochs=100

)



model.fit(X, y)



predictions = model.predict(X)



print("Predictions:", predictions)

print("Weights:", model.weights)

print("Bias:", model.bias)

```



The expected predictions after successful training are:



```text

Predictions: [0 0 0 1]

```



The exact learned weights and bias are not unique. Different parameter values can implement the same Boolean function.



## 17.2 Understanding the code



### Initialization



```python

self.weights = np.zeros(n_features)

self.bias = 0.0

```



All weights and the bias start at zero.



### Forward pass



```python

z = np.dot(x, self.weights) + self.bias

```



This calculates:



$$

z=\mathbf{w}^{T}\mathbf{x}+b

$$



### Prediction



```python

return 1 if z >= 0 else 0

```



This applies the step function.



### Error calculation



```python

error = target - prediction

```



This implements:



$$

e=y-\hat y

$$



### Weight update



```python

self.weights += self.learning_rate * error * xi

```



Mathematically:



$$

\mathbf{w}\leftarrow\mathbf{w}+\eta e\mathbf{x}

$$



### Bias update



```python

self.bias += self.learning_rate * error

```



Mathematically:



$$

b\leftarrow b+\eta e

$$



### Stopping condition



```python

if mistakes == 0:

    break

```



If an entire epoch contains no misclassified training examples, the training loop stops.



If the dataset is not linearly separable, the model may continue making mistakes until the maximum epoch count is reached.



---



# 18. Perceptron Using Scikit-learn



Scikit-learn provides a production-quality implementation of the classic linear perceptron classifier.



```python

from sklearn.linear_model import Perceptron

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X = [

    [0, 0],

    [0, 1],

    [1, 0],

    [1, 1]

]



y = [0, 0, 0, 1]



model = make_pipeline(

    StandardScaler(),

    Perceptron(

        max_iter=1000,

        eta0=0.1,

        random_state=42

    )

)



model.fit(X, y)



predictions = model.predict(X)



print("Predictions:", predictions)

print(

    "Accuracy:",

    accuracy_score(y, predictions)

)

```



Expected classification:



```text

Predictions: [0 0 0 1]

Accuracy: 1.0

```



Here, accuracy is evaluated on the same four observations used for training, so it demonstrates that the model can represent the AND function. It is not an independent estimate of performance on unseen data.



For a real dataset, split the data into training and test sets before fitting the model. A pipeline helps ensure that the scaler is fitted only on the training portion.



The scikit-learn implementation also supports options such as regularization, shuffling, and multiclass classification.



---



# 19. Advantages and Limitations



## 19.1 Advantages



### 1. Simplicity



The model is easy to understand mathematically and implement in code.



### 2. Computational efficiency



Its basic operations consist of weighted sums, thresholding, and parameter updates.



### 3. Online learning



The classic update rule can process examples one at a time rather than requiring all observations to be processed simultaneously.



### 4. Interpretability



The learned weights and bias directly define a linear decision boundary.



### 5. Theoretical guarantees



The perceptron convergence theorem provides a finite-mistake guarantee for linearly separable training data under standard assumptions.



## 19.2 Limitations



### 1. Linear decision boundary



A single perceptron cannot represent arbitrary nonlinear decision boundaries in the original feature space.



### 2. XOR limitation



A single perceptron cannot correctly classify all XOR examples.



### 3. No probability output



The hard step function produces a class label, not a calibrated probability.



### 4. No general convergence guarantee



If the training data are not linearly separable, the standard perceptron may continue making mistakes.



### 5. Sensitivity to data representation



Poorly scaled features, noisy labels, and unsuitable features can reduce practical performance.



### 6. Limited expressive power



A single perceptron is not suitable for many complex tasks involving intricate image patterns, language, or nonlinear relationships.



### 7. The result depends on training conditions



The order of examples, initialization, learning rate, and stopping rule can affect the parameters obtained.



---



# 20. Perceptron vs. Logistic Regression vs. Multilayer Neural Network



| Property | Single perceptron | Logistic regression | Multilayer neural network |

|---|---|---|---|

| Basic computation | Weighted sum and threshold | Weighted sum and sigmoid | Multiple layers of weighted computations |

| Typical output | Binary class | Probability and class | Task-dependent output |

| Common activation | Step | Sigmoid for binary classification | ReLU, sigmoid, tanh, softmax, etc. |

| Standard training | Mistake-driven updates | Gradient-based optimization | Backpropagation and optimization |

| Linear boundary in original features | Yes | Yes for the usual linear model | Can learn nonlinear boundaries |

| Direct probability estimation | No | Yes, though calibration is not guaranteed | Depends on output and training |

| Can represent XOR with the usual architecture? | No | No, without feature transformations | Yes, with a suitable architecture |

| Complexity | Low | Low | Potentially high |



## 20.1 Perceptron vs. logistic regression



A perceptron uses a hard threshold:



$$

\hat y=\mathbb{1}[z\geq0]

$$



Logistic regression uses the sigmoid function:



$$

\sigma(z)=\frac{1}{1+e^{-z}}

$$



and estimates:



$$

P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^{T}\mathbf{x}+b)

$$



The sigmoid maps real-valued scores to values strictly between 0 and 1.



Logistic regression is commonly trained by minimizing binary cross-entropy.



For a target $y\in\{0,1\}$ and predicted probability $p$:



$$

\boxed{

L=-[y\log p+(1-y)\log(1-p)]

}

$$



This loss penalizes incorrect probability estimates, not merely incorrect class labels.



## 20.2 Perceptron vs. multilayer neural network



A single perceptron computes one linear score followed by an activation.



A multilayer neural network combines many artificial neurons across layers.



For example, a one-hidden-layer network can be expressed as:



$$

\mathbf{h}=\phi(W_1\mathbf{x}+\mathbf{b}_1)

$$



$$

\hat y=\psi(W_2\mathbf{h}+\mathbf{b}_2)

$$



where:



- $W_1,W_2$: weight matrices.

- $\mathbf{b}_1,\mathbf{b}_2$: bias vectors.

- $\phi,\psi$: activation functions.

- $\mathbf{h}$: hidden representation.



The hidden layer can transform the input into a representation that supports a more complex decision boundary.



Modern deep learning develops this basic idea into networks with many layers and specialized architectures.



---



# 21. Historical Significance



The perceptron is an important milestone in the history of artificial intelligence.



- In 1943, Warren McCulloch and Walter Pitts published an influential mathematical model of an artificial neuron.

- In 1957–1958, Frank Rosenblatt developed and publicized the perceptron learning model.

- In 1969, Marvin Minsky and Seymour Papert's book *Perceptrons* analyzed important limitations of perceptron models, including the inability of a single-layer perceptron to represent XOR.

- Later advances in multilayer networks and learning algorithms helped establish the foundations of modern neural-network research.



The historical lesson is not that perceptrons are useless. It is that a simple model can have powerful theoretical properties while still possessing fundamental representational limitations.



---



# 22. Common Interview Questions



## Q1. What is a perceptron?



A perceptron is a simple artificial neuron that calculates a weighted sum of input features, adds a bias, and applies an activation function to produce a prediction.



## Q2. What is the mathematical equation of a perceptron?



$$

z=\mathbf{w}^{T}\mathbf{x}+b

$$



$$

\hat y=f(z)

$$



## Q3. What is the purpose of weights?



Weights determine the contribution of each input feature to the weighted sum.



## Q4. What is the purpose of bias?



Bias shifts the decision boundary and allows the model to classify inputs without forcing the boundary through the origin.



## Q5. What activation function is used in a classic perceptron?



A step or threshold function is traditionally used.



## Q6. What is the perceptron learning rule?



For labels 0 and 1:



$$

w_i\leftarrow w_i+\eta(y-\hat y)x_i

$$



$$

b\leftarrow b+\eta(y-\hat y)

$$



## Q7. Why does the perceptron fail on XOR?



XOR is not linearly separable. No single straight-line boundary can separate all positive examples from all negative examples.



## Q8. What is linear separability?



A dataset is linearly separable if a linear decision boundary can correctly separate the classes.



## Q9. Does the perceptron always converge?



No. The classic finite-mistake convergence guarantee applies to linearly separable training data under the relevant assumptions.



## Q10. Does a perceptron perform gradient descent?



The classic perceptron algorithm is a mistake-driven update method. It is not ordinary gradient descent on binary cross-entropy through a differentiable step activation.



## Q11. Can a perceptron solve AND and OR?



Yes. Both are linearly separable Boolean functions.



## Q12. Can a perceptron solve XOR?



A single standard perceptron cannot. A suitable multilayer network can.



## Q13. What is the difference between a perceptron and a neuron in a deep neural network?



A perceptron traditionally uses a hard threshold and a mistake-driven update. Artificial neurons in modern networks commonly use differentiable activations and are trained with backpropagation and gradient-based optimization.



## Q14. What is the decision boundary of a perceptron?



$$

\mathbf{w}^{T}\mathbf{x}+b=0

$$



It is a hyperplane in the input feature space.



## Q15. Why is the step activation unsuitable for ordinary backpropagation?



It is constant almost everywhere and nondifferentiable at the threshold. Its ordinary derivative therefore does not provide a useful gradient for standard backpropagation.



## Q16. What happens if all weights and the bias are initialized to zero?



The initial score is zero for every input, so under the convention used here the model initially predicts class 1 for every observation. Subsequent mistakes can change the parameters.



## Q17. Is the perceptron a regression algorithm?



The classic perceptron is a classification algorithm. It outputs class labels rather than continuous regression estimates.



## Q18. Can a perceptron have more than two inputs?



Yes. It can have any finite number of input features.



## Q19. Is a perceptron a neural network?



A single perceptron can be considered a one-neuron neural model. A multilayer neural network contains interconnected computational units arranged in layers.



## Q20. What is the most important limitation of a single perceptron?



Its decision boundary is linear in the input features, so it cannot represent every classification problem.



---



# 23. Practice Problems



Try solving these problems before looking at the answers.



## Beginner Level



### Problem 1



Given:



$$

x_1=2,\quad x_2=3

$$



$$

w_1=4,\quad w_2=2,\quad b=-5

$$



Calculate the weighted sum and prediction.



### Problem 2



What is the purpose of a bias in a perceptron?



### Problem 3



If the weighted sum is $-4$, what is the output under the step-function convention used in these notes?



### Problem 4



If the actual label is 0 and the predicted label is 0, what is the error?



### Problem 5



What type of classification boundary does a single perceptron produce?



## Intermediate Level



### Problem 6



A perceptron has:



$$

w=2,\quad b=-4

$$



Find its decision boundary for a single input feature.



### Problem 7



Given:



$$

x_1=2,\quad x_2=1

$$



$$

w_1=1,\quad w_2=-2,\quad b=0

$$



Calculate the prediction.



### Problem 8



Suppose:



$$

x=3,\quad y=1,\quad\hat y=0

$$



$$

w=0.5,\quad b=-1,\quad\eta=0.1

$$



Calculate the updated weight and bias.



### Problem 9



Why can the perceptron represent AND but not XOR?



### Problem 10



What happens if a perceptron is trained on non-linearly separable data?



## Advanced Level



### Problem 11



Derive the decision boundary:



$$

3x_1-2x_2+6=0

$$



Express $x_2$ in terms of $x_1$, and identify the slope and intercept.



### Problem 12



Explain why the classic perceptron convergence theorem requires linear separability.



### Problem 13



Show how to write the perceptron equation in augmented-vector form.



### Problem 14



Explain the difference between the 0/1-label update rule and the signed-label update rule.



### Problem 15



Explain why the perceptron update rule is not the same as gradient descent on binary cross-entropy.



---



# 24. Practice Problems: Answers and Explanations



## Answer 1



$$

z=(4)(2)+(2)(3)-5

$$



$$

z=8+6-5=9

$$



Since $z\geq0$:



$$

\boxed{\hat y=1}

$$



## Answer 2



Bias shifts the decision boundary and allows it to move away from the origin.



## Answer 3



$$

z=-4<0

$$



Therefore:



$$

\boxed{\hat y=0}

$$



## Answer 4



$$

e=y-\hat y=0-0=0

$$



No update is needed.



## Answer 5



A linear decision boundary: a point for one input, a line for two inputs, a plane for three inputs, and a hyperplane for higher dimensions.



## Answer 6



The score is:



$$

z=2x-4

$$



At the boundary:



$$

2x-4=0

$$



$$

\boxed{x=2}

$$



## Answer 7



$$

z=(1)(2)+(-2)(1)+0

$$



$$

z=2-2=0

$$



Because zero is assigned to class 1:



$$

\boxed{\hat y=1}

$$



## Answer 8



The error is:



$$

e=1-0=1

$$



Weight update:



$$

w_{\text{new}}=0.5+(0.1)(1)(3)

$$



$$

\boxed{w_{\text{new}}=0.8}

$$



Bias update:



$$

b_{\text{new}}=-1+(0.1)(1)

$$



$$

\boxed{b_{\text{new}}=-0.9}

$$



## Answer 9



AND is linearly separable: one straight-line boundary can separate its positive example from its negative examples.



XOR is not linearly separable: no single straight-line boundary can correctly separate all four examples.



## Answer 10



The perceptron may continue making mistakes because no single linear boundary correctly separates all the examples. Training can stop at a predefined epoch limit, but that does not imply that a perfect classifier has been found.



## Answer 11



Start with:



$$

3x_1-2x_2+6=0

$$



Rearrange:



$$

-2x_2=-3x_1-6

$$



Divide by $-2$:



$$

\boxed{x_2=\frac{3}{2}x_1+3}

$$



The slope is:



$$

\boxed{m=1.5}

$$



The vertical-axis intercept is:



$$

\boxed{c=3}

$$



## Answer 12



The convergence theorem relies on the existence of a separating hyperplane. If the data are not linearly separable, the mistake-driven updates need not reach parameters that correctly classify every training example.



## Answer 13



Define:



$$

\tilde{\mathbf{x}}=

\begin{bmatrix}

\mathbf{x}\\1

\end{bmatrix}

,\qquad

\tilde{\mathbf{w}}=

\begin{bmatrix}

\mathbf{w}\\b

\end{bmatrix}

$$



Then:



$$

\boxed{z=\tilde{\mathbf{w}}^{T}\tilde{\mathbf{x}}}

$$



## Answer 14



The 0/1 formulation uses:



$$

e=y-\hat y

$$



and updates:



$$

\mathbf{w}\leftarrow\mathbf{w}+\eta e\mathbf{x}

$$



The signed-label formulation uses $y\in\{-1,+1\}$ and updates a misclassified example using:



$$

\mathbf{w}\leftarrow\mathbf{w}+\eta y\mathbf{x}

$$



The two formulations encode the classes differently and should be used consistently.



## Answer 15



The classic perceptron updates parameters based on classification mistakes and uses a hard threshold. Binary cross-entropy is a differentiable probabilistic loss typically used with logistic regression or a neural network's sigmoid output. The optimization objectives and update derivations are different.



---



# 25. Summary and Key Formulas



## 25.1 The fundamental model



$$

\boxed{z=\mathbf{w}^{T}\mathbf{x}+b}

$$



$$

\boxed{\hat y=f(z)}

$$



## 25.2 Step activation function



$$

\boxed{

f(z)=

\begin{cases}

1 & z\geq0\\

0 & z<0

\end{cases}

}

$$



## 25.3 Decision boundary



$$

\boxed{\mathbf{w}^{T}\mathbf{x}+b=0}

$$



## 25.4 Error for binary labels



$$

\boxed{e=y-\hat y}

$$



## 25.5 Weight update



$$

\boxed{w_i\leftarrow w_i+\eta e x_i}

$$



## 25.6 Bias update



$$

\boxed{b\leftarrow b+\eta e}

$$



## 25.7 Vector update



$$

\boxed{\mathbf{w}\leftarrow\mathbf{w}+\eta e\mathbf{x}}

$$



## 25.8 Signed-label update



For a misclassified example with $y\in\{-1,+1\}$:



$$

\boxed{\mathbf{w}\leftarrow\mathbf{w}+\eta y\mathbf{x}}

$$



The bias is updated as:



$$

\boxed{b\leftarrow b+\eta y}

$$



## 25.9 Logistic regression comparison



$$

\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}

$$



## 25.10 Perceptron convergence bound



Under the standard assumptions of the perceptron convergence theorem:



$$

\boxed{M\leq\left(\frac{R}{\gamma}\right)^2}

$$



---



# Final Conceptual Understanding



A perceptron is much more than a formula involving weights and bias. It introduces several foundational ideas in machine learning:



1. **Representation:** Expressing a classification problem using numerical features.

2. **Parameters:** Learning weights and a bias from examples.

3. **Decision-making:** Converting a numerical score into a class.

4. **Geometry:** Understanding classification through linear decision boundaries.

5. **Learning:** Updating parameters when predictions are incorrect.

6. **Generalization:** Recognizing that success on training data does not automatically guarantee success on unseen data.

7. **Model limitations:** Understanding why some problems require more expressive models.

8. **Neural-network foundations:** Connecting the single artificial neuron to multilayer networks.



The most important equation to remember is:



$$

\boxed{\hat y=f(\mathbf{w}^{T}\mathbf{x}+b)}

$$



The most important learning rule to remember is:



$$

\boxed{\mathbf{w}\leftarrow\mathbf{w}+\eta(y-\hat y)\mathbf{x}}

$$



And the most important limitation to remember is:



**A single perceptron can learn only a linear decision boundary in its input feature space.**



Understanding these three ideas gives you a strong foundation for studying activation functions, loss functions, gradient descent, backpropagation, artificial neural networks, and deep learning.
