# Introduction

Over this past year, I've committed myself to the dense reading of "Neural Networks and Deep Learning" by Michael Nielsen. This project encompasses the many hours I've spent diving deep into the math and low-level logic of neural networks.

# Cross-Entropy vs Quadratic Cost
Nielsen's deep neural network allows the user a choice between turning on either a quadratic cost or a cross-entropy cost function.

## Quadratic Cost (MSE)
This is the quadratic cost function where y is the actual and a is the predicted.

![](img/quadratic/qc-equation.png)

For our gradient descent algorithm, we must use the derivative of the quadratic cost function with respect to the weights and biases like so:

![](img/quadratic/qc-derivatives.png)

Since this derivative relies on the sigmoid function, let's analyze it.

![](img/quadratic/sigmoid_function.png)

Quadratic cost is great for training nodes that have a weighted input (z) that's close to 0. In this case, during gradient descent, the derivative of the sigmoid function is very high, resulting in large positive changes to the overall network's cost. However, trouble arises when the weighted input is either very incorrect or correct. In this case, the derivative of the sigmoid function (quadratic cost) is very small, and we'll only see very small changes during gradient descent. For example, in the sigmoid graph above, we can see that when z is near -4 or 4, the slope plateaus, resulting in a smaller derivative. Conversely, when z is around zero, the slope is steep and we see large changes in our cost. This slow learning can severely affect the rate at which our neural network learns.

## Cross-Entropy Cost
Our second choice of a cost function is the cross-entropy function.

![](img/cross_entropy/ce-eq.png)

In this equation, *n* is the total number of items in the training data (*n* handwritten digits), *x* is each training image, *y* is the actual, and *a* is the predicted. A benefit to the cross-entropy is that it avoids the issue of learning slowdown that quadratic cost has.

![](img/cross_entropy/ce-deriv1.png)
![](img/cross_entropy/ce-deriv2.png)

In both of these partial derivatives, we are no longer dependent on the derivative of the sigmoid function like in the quadratic cost function. Instead, in these, like one would expect, the cost is dependent on how large the error is (predicted - actual). For the rest of this project, we will be solely working with the cross-entropy function.


# Training Epochs
Initially, this deep neural network was set to an arbitrary 30 epochs. The results of 30 cycles on the training data are displayed in Figure 1.

![](img/cross_entropy/epoch29_output.png)

Figure 1

<!-- ctrl + shift + v for markdown preview 
talk about sigmoid vs cross entropy cost
talk about lambda affects and targeting small weights
-->

For a visualization of this, we can see the cost on the training data in Figure 2. It slopes down dramatically between epochs 0 through 10. After that, the amount of cost reduction versus time and computational power severely decreases. Interestingly, it seems the graph takes on a y=1/x shape.

![](img/cross_entropy/cost_on_training_data.png)

Figure 2

As we can see here, specifically it looks like after epoch 5, we see the accuracy rate of change severely drop off. This phase of training wastes a lot of computing power and time and is often associated with overfitting.

![](img/cross_entropy/Accuracy_training_data.png)

Figure 3

It does appear that the network continues to trend up, so here one would have to make a choice of valuing computer time/power vs how accurate their network becomes. If computing time is a concern, one might feel okay with keeping their neural network at 80% accuracy and stopping at epoch 5-8. If computing time is not a concern, then keeping the network at 30 epochs to obtain the last 5% would be the best route.

