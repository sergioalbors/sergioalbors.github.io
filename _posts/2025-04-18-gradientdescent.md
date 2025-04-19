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

                        $\{2x^2}-5=0$

We could clearly see its minima is $4x=0$

But with much more complex functions, which depend on more than 1 variable this wont' do it. 
To apply the gradient descent method, there is 2 things we need. 

    - the direction
    
    - learning rate

## How to know the direction of more complex functions?


Para encontrar la direccion de máximo crecimiento de funciones más complejas que dependen de más de una variable, es necesario derivar primero respecto a una variable, luego respecto a otra, y así sucesivamente. Por ejemplo, si nuestra función de coste depende de tres variables, tendremos que calcular tres derivadas parciales, una para cada variable.

Las derivadas parciales permiten calcular el gradiente de una función. El gradiente es un vector que apunta en la dirección del crecimiento máximo de la función y nos indica cómo cambian los valores de la función en cada dirección.

Por ejemplo, si nuestra función coste depende de 2 variables como son x e y, 

$C(x,y)=2x-3y+6$

Deberemos derivar primero respecto a x, y luego respecto a y quedando asi:

$\frac{\partial C}{\partial x}=2$

$\frac{\partial C}{\partial y}=-3$

Por lo tanto el gradiente de nuestra función coste será:

$\nabla f = \left( 2, -3 \right)$

Para funciones mucho más complejas se inicializa una variable a un valor al azar antes de derivar respecto a ella.

## Learning rate
Lo que hace gradient descent, es coger la dirección de máximo crecimiento e ir hacia la dirección contraria ya que lo que queremos es minimizar la función coste, o en otras palabras minimizar el error entre nuestro valor predecido y el actual. Una vez tenemos el gradiente con los valores iniciales al azar. Gradient descent lo que hace es va avanzando a pasos muy pequeños a la dirección contraria del gradiente. Ya que como hemos dicho queremos minimizar y el gradiente nos da el máximo crecimiento, por lo tanto, lo contrario. 




