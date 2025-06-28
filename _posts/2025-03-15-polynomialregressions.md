---
layout: post
title: Polynomial regressions
date: 2025-03-15
tags: regressions lasso,ridge
description: polynomial regressions explained
mermaid:
  enabled: true
  zoomable: true
---

## GETTING STARTED

This page is written to teach people or give an idea about what regressions are used for, there mathematical baground and it's appliances in our day to day. You can find the python implementation code of everything talked about below the theory explanations.

A regression is a mathematical analysis method used to predict the outcome of future events between other things. What it does, it estimates the relationship between a dependent variable and one or more independent variables. This can be used to make predictions of the future.

The simplest example:

If you know the number of study hours of a student, you can predict his score on a test.

### WHAT IS POLYNOMIAL REGRESSION?

A polynomial regression is a form of regression analysis in which the relation between the independent variables and dependent variables are modeled in a n degree polynomial.

This polynomials can go from a 1st degree polynomial to a nth degree polynomial:

linear:- $y = a0 + a1x$

quadratic:- $y = a0 + a1x + a2x^2$

nth grade:- $y = a_0 + a_1 x + a_2 x^2 + \dots + a_n x^n$

With this being said, we suppose our equation has n degrees and we want to minimize the error between the training data and the predicted values, ideally we'd like our error to be 0, this would mean we've built a regression model with 100% accuracy or that our function doesn't have any gausian noise.

So the we are seeking to minimize ( get as close to 0 as we can) the following:

$$
E = \sum_{i=1}^m \left( y_i - \hat{y}(x_i) \right)^2
$$

### Why is polynomial regression so important?

Let's consider a case of simple linear regression:

linear vs polynomial

![alt text](/assets/img/img/linearvspolynomial.png)

As we can see in the picture above, the linear model has very poor performance, whereas the polynomial model has a much better adjustment and consequently, will have a lower error.

Polynomial regression is used when the relationship between our data samples isn't ineal, and consequently the data samples form a kind of curve or multiple curves that cannot be fitted with a straight line.

### VANDERMONDE(Square system woth n points --- Polynomial of degree n-1)

We will only be able to apply this method if the xi are different and our polynomial is a degree less than the number of points (n-1), then our matrix will be invertible.

#### VAMDERMONDE EXPLANATION

1. We have our number of data points, could be 10 could be n points.
2. We assume the equation that best fits the points has this form:

$f(x) = a0 + a1x + a2x^2 + a3x^3 + ... + anx^n + e$

3. With this information, we can proceed to form our vandermonde matrix to find the coefficients of this polynomial.

$$
V(a_0, a_1, \dots, a_{n-1}) =
\begin{bmatrix}
1 & a_0 & a_0^2 & \cdots & a_0^{n-1} \\
1 & a_1 & a_1^2 & \cdots & a_1^{n-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & a_{n-1} & a_{n-1}^2 & \cdots & a_{n-1}^{n-1}
\end{bmatrix}
$$

4. The equation we'll have to solve has the following form:
   $VA=Y$

where :

- V: our vandermonde matrix

- A: matrix of coefficients

- Y: output vector, the y axis of our points

5. If we want to figure out the values of the coefficients of our polynomial, then what we want is the X, the matrix of coefficients.

Solving the previous equation would lead to:
$VA=Y$ =

= $V^-1Y=A$

Where $V^-1$ = the inverse of our vandermonde matrix

The inverse of a matrix is:

$$
V^{-1} = \frac{\operatorname{adj}(V^T)}{\det(V)}
$$

The determinant of our vandermonde matrix is:

$$
\det(V) =
\prod_{1 \le i < j \le n} (x_j - x_i)
$$

Once we have the coefficients of our polynomial, we will have found the equation that best fits our data samples. Here is an example implementeed in python code:

Let's say we want to create a function that best fits our 20 points, so our function will be a 19th degree polynomial.

import numpy as np

import matplotlib.pyplot as plt

n = 10

x = np.random.uniform(-100, 100, n)

y = np.random.uniform(-100, 100, n)

a = np.polyfit(x, y, n-1)

V = np.vander(x, N=len(x), increasing = False) # we create our vandermonde matrix

v\*a=y # we represent our system of linear equations

a = np.linalg.solve(V, y) # we solve our system

polinomio = np.poly1d(a) # a = coefficients of our polynomial


x_fit = np.linspace(min(x), max(x), 100) # we generate a set of points to plot

y_fit = polinomio(x_fit)

plt.scatter(x, y, color = 'red')

plt.plot(x_fit, y_fit)

plt.show()

print(polinomio)

Here is the result in pyhton:

![alt text](/assets/img/img/vandermonde2.png)

and the polynomial which is shown at the terminal ( if there is no terminal you can press ctrl + j):

![alt text](/assets/img/img/polynomial.png)

### (X TX)^−1X^TY --- More points than coefficients

#### OVERFITTING VS UNDERFITTING

The first thing we need to have clear is what is considered in the ml world as a good predicting model:
if our model does the following things it can be considered good:

- our model avoids underfitting and overfitting

- adapts well to new/unseen data

- brings the error/cost function very close to 0 (closely matches real value with the predicted one)

Let's get into what the first point means and how can we avoid that.

##### Overftting

Overfitting appears when our model has too many parameters and learns too much from the training data, including learning from details that aren't relevant, like noise.
This model fits very well our training data samples but sticks too much to them failing to make a good prediction of the new data.

See it as a student that prepares for a test memorizing the answers of the last exam without understanding the topic, it will do very good on the exam he has memorized, but in the actual exam he will get a very low score because he sticks too much to the training data.

##### Underfitting

Underfitting is the opposite of overfitting. Instead of being too complex, underfitting appears when a model is too simple to capture what is really going on with the data.

If we tried to do a linear regression on data that forms a curve, our prediction line, would fit awfully our points right? The line would miss a lot of points.

If a student doesn't study at all, he will score poorly both on the actual exam and the practice exams.

ways to avoid underfitting:

- increase complexity of our prediction

![alt text](/assets/img/image.png)

### Regularization less points than coefficients -- Ridge/Lasso

A very useful way too fight or overcome overfitting problems in our models is to add a regularization. There are plenty of types but we will cover the 2 principal regularizations, L2 and L1.

Overfitting is a problem that occurs when the regression model gets tuned to the training data too much that it does not generalize well. It is also called a model with high variance as the difference between the actual value and the predicted value of the dependent variable in the test set will be high.

LASSO (L1)

Lasso helps us fight overfitting by penalizing certain coefficients and shrinking them towards zero. This penalty "targets" coefficients that are least important and just add noise to our model.

For instance, imagine we are packing for a trip to vietnam and we are carrying only one bag and the company you are flying with doesn't allow hand baggage to weigh more than 8 kg and after packing, you find out your bag is 3 kg overweight. Eventually, you'll have to get rid of things until you meet with the weight requirements of the airline, but there are things you need and things that aren't that important. So to reduce weight you'll have to get rid of the things thst aren't crucial for your trip.

Lasso does the same thing, targets features that aren't really important for our model and shrinks them towards 0 to make them irrelevant. If your model didn't shirnk them towards 0, it wouldn't be very accurate due to your prediction trying to fit every single variable and would end up highly overfitted resulting in a very inaccurate model.

In order to add a type L1 regularitzation to our model, first we'll have to understand a new concept called: Gradient descent. You can find an explanation based on what I've understood [here](https://sergioalbors.github.io/blog/2025/gradientdescent/).

RIDGE (L2)

If we drop in variance, our line will not fit the training data as good as a linear regression line, but it provides better long term predictions. We sacrifice good adaptation to the training data in order to get a better fit on the predicited values.

Let's compare the simple regression model matrix equation with the ridge regression model matrix equation to see how they differ from each other:

$$
w = (X^T X + \lambda I)^{-1} X^T y
$$

this being the penalty added equation

$$
w = (X^T X)^{-1} X^T y
$$

this being a simple linear regression

Very similarly, the ridge model seeks the same thing but with a new concept added to the equation, something called penalty.
Ridge regression adds a regularization term that helps our model prevent overfitting by penalizing large coeffcicients nad sticking to smaller ones in order to stabilize the model.

Let's see an example implemented in python code:

import numpy as np

import matplotlib.pyplot as plt

n = 100

np.random.seed(42)

x = np.random.uniform(-2, 2, n)

x = np.sort(x)

def f(x): # the function we'll be using

return np.sin(2*x)

y = f(x)

y_t = f(x) + 10*np.random.randn(n) # we add some gaussian noise to our function

degree = 4

X = np.vstack([x**i for i in range(degree + 1)]).T # create our matrix

alpha = 10 # the "severeness" of our penalty

I = np.eye(X.shape[1]) # we form our matrix n x n

I[0, 0] = 0 # we avoid penalizing the constant value of our model

r4 = np.matmul(np.linalg.inv(np.matmul(X.T, X) + alpha \* I ),np.matmul(X.T, y_t)) # we solve the equation mentioned above

yh4 = np.matmul(X, r4) # multiply our matrix by the result of the equation above to get the coefficients

plt.scatter(x, y, label = 'data')

plt.plot(x, yh4, color = 'red', label = 'predicted values')

plt.legend()

plt.show()

### Choosing the correct value for lambda

At first, the value we have to establish $\lambda\$ at is not known and a way to choose the correct value is to test many values and see which one performs best. However there are multiple algorithm that can help us determnine the most appropriate value of this parameter for each case.

This value is crucial to accomploish a succesfull/accurate regression model. This parameter controls how strong our coefficientrs will be shrunk towards 0. If $\lambda\=0$ our penalty dissapears and we will only be minimizing a simple regression model. And the larger we make $\lambda\$ our predictions for y will become less sensitive to s, due to the slope of our line getting asymptotically closer to 0.
