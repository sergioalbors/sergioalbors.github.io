---
layout: post
title: Polynomial regressions
date: 2025-03-15
tags: formatting diagrams
description: an example of a blog post with diagrams
mermaid:
  enabled: true
  zoomable: true
---

## GETTING STARTED

This page is written to teach people or give an idea about what regressions are used for, there mathematical baground and it's appliances in our day to day. 

A regression is a mathematical analysis method used to predict the outcome of future events between other things. What it does, it estimates the relationship between a dependent variable and one or more independent variables. This can be used to make predictions of the future. 

 The simplest example: 

 If you know the number of study hours of a student, you can predict his score on a test. 

### WHAT IS POLYNOMIAL REGRESSION?

A polynomial regression is a form of regression analysis in which the relation between the independent variables and dependent variables are modeled in a n degree polynomial. 

This polynomials can go from a 1st degree polynomial to a nth degree polynomial:

linear:-               $y = a0 + a1x$

quadratic:-         $y = a0 + a1x + a2x**2$

nth grade:-       $y = a0 + a1x + a2x**2 + ... +  anx**n$

Why is polynomial regression so important?

Let's consider a case of simple linear regression: 

IMAGEIMAGEIMAGEIMAGEsn vabpidàn`dunaO`<

linear vs polynomial


As we can see in the picture above, the linear model has very poor performance, whereas the polynomial model has a much better adjustment and consequently, will have a lower error. 

Polynomial regression is used when the relationship between our data samples isn't ineal, and consequently the data samples form a kind of curve or multiple curves that cannot be fitted with a straight line. 



### VAMDERMONDE EXPLANATION

1. We have our number of data points, cpuld be 10 could n points. 
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












