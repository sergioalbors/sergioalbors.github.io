---
layout: post
title: GRADIENT DESCENT
date: 2025-04-18
tags: gradientdescent
description: a basic explanation about gradient descent and it's possible applications
mermaid:
  enabled: true
  zoomable: true
---

## Getting started

After reading this post, you will hopefully understand the concept of gradient descent, the maths behind it and how can it be applied for a lasso regression to predict the outcome of future events. The person who wrote this blog me:) , assumes the reader has basic calculus knowledge such as the concept of derivatives, maxima, minima and the slope of a function and other things.

Gradient descent is an algorithm used to estimate the values of the parameters of a function used to reduce a cost function, or the error between our predicted values and the actual values.

Similar to the least squares method, we want to minimize our error/cost function to keep the error between our predictions and the actual values as close as 0 as we can. However to do so, we cannot apply the basic calculus method of:

$f'(x)=0$

If we had a simple polynomial such as a quadratic education like:

${2x^2}-5=0$

We could clearly see its minima is $4x=0$

But with much more complex functions, which depend on more than 1 variable this wont' do it.
Let's see an example to prove why we can't apply the $f'(x)=0$ rule:

Let's assume we have a function that depends on 2 variables such as:

$f(x,y)=x^2+y^2+2xy$

If we do the regular method($f'(x)=0$), we'd have $2x+2y=0$, which is equivalent to $x=-y$. There is so much information missing here, this only shows that there exists a relationship between $x$ and $y$, but we don't know how this function changes with respect to $y$.

The equation that describes what this algorithm does is:

$\mathbf{d} = \mathbf{a} - \lambda \nabla f(\mathbf{a})$

where:

- a = our learning rate

- $\nabla f(a)$ = the gradient of our function

In order to solve complex functions with these equation, will have to figure out what the following things mean:

##### 1. the direction (gradient)

##### 2. the learning rate

## 1.How to know the direction of more complex functions?

To find the direction of maximum growth for more complex functions that depend on more than one variable, it is necessary to first differentiate with respect to one variable, then with respect to another, and so on. For example, if our cost function depends on three variables, we will need to calculate three partial derivatives, one for each variable.

Partial derivatives allow us to compute the gradient of a function. The gradient is a vector that points in the direction of the function's maximum growth and indicates how the function's values change in each direction.

For instance, if our cost function depends on 2 variables like $x$ or $y$,such as:

$C(x,y)=2x-3y+6$

We will have to first differentiate with respect to $x$, and then differentiate with respect to $y$, resulting in:

$\frac{\partial C}{\partial x}=2$

$\frac{\partial C}{\partial y}=-3$

Therefore, the gradient of our cost function will be:

$\nabla f = \left( 2, -3 \right)$

Once we know the direction of maximum growth, (2, -3) in this case, gradient descent works by taking very small steps in the opposite direction of the gradient. This small steps are the learning rate of our cost function.

###### Why the opposite direction?

Because we want to find the minima, and the gradient as we stated multiple times, gives us the maximum growth, so the opposite direction will hopefully lead us to the lowest point of the function (spoiler: if we don't skip that point).

## 2.Learning rate

The Learning rate determinates the size of the little jumps forward or steps taken in the opposite site of the gradient to reach the minimum. The learning rate sometimes called alpha usually has a small value.

If the alpha is too high, the steps are bigger, but there is a potential risk of overshooting the minimum point and skipping it.

However, a very small learning rate isn´t ideal either because it takes too many steps to get to the lowest point of the cost function and this could compromise the overall efficiency of your algorithm.

Think at it as if you were trying to sintonize a radio to a certain frequence. Let's say we start at a frequency of 2.50 hz, and the frequence of our unit is at 15.50hz, if you start adding 5 hz every time, you will never find the right frequency due to your learning rate being too high and missing your target. But if you add little steps of 0.1 hz, it could take you forever to find the right frequence.

![alt text](/assets/img/learningrate.png)