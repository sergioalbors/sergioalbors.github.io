---
layout: post
title: GRADIENT DESCENT
date: 2025-04-18
tags: formatting diagrams
description: a basic explanation about gradient descent and it's possible aplications
mermaid:
  enabled: true
  zoomable: true
---
## Getting started


After reading this post, you will hopefully understand the concept of gradient descent, the maths behind it  and how can it be applied for a lasso regression to predict the outcome of future events. The person who wrote this blog me:) , assumes the reader has basic calculus knowledge such as the concept of derivatives, maxima, minima and the slope of a function and other things. 

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

If we do the regular method($f'(x)=0$), we'd have $2x+2y=0$, which is equivalent to $x=-y$ but this doesn't mean that x=0 minimizes the function. There is so much information missing here, this only shows that there exists a relatinoship between $x$ and $y$, but we don't know how this function changes with respect to $y$.

In order to solve more complex functions, will have to figure out two things:
### 1. the direction

### 2. the learning rate

## How to know the DIRECTION of more complex functions?


To find the direction of maximum growth for more complex functions that depend on more than one variable, it is necessary to first differentiate with respect to one variable, then with respect to another, and so on. For example, if our cost function depends on three variables, we will need to calculate three partial derivatives, one for each variable.

Partial derivatives allow us to compute the gradient of a function. The gradient is a vector that points in the direction of the function's maximum growth and indicates how the function's values change in each direction.

Por ejemplo, si nuestra función coste depende de 2 variables como son x e y, 

$C(x,y)=2x-3y+6$

Deberemos derivar primero respecto a x, y luego respecto a y quedando asi:

$\frac{\partial C}{\partial x}=2$

$\frac{\partial C}{\partial y}=-3$

Por lo tanto el gradiente de nuestra función coste será:

$\nabla f = \left( 2, -3 \right)$


Once we know the direction of maximum growth, (2, -3) in this case, gradient descent works by taking very small steps in the opposite direction of the gradient. This small steps are the learning rate of our cost function.  

## Learning rate






