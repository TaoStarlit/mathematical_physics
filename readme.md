# coding:utf-8
#max_likelihood_estimation.py
# MLE: find the parameter values that maximize the likelihood of making the observations given the parameters.
linear model and normal distribution,
eg. find the coefficients k and intercept b, and the mean mu and σ sigma square deviation
understand the condition probility, trainin       g set, lable(or not)

#the model is to describe the distribution:
*samples of idd observation <-- a distribution  with an unknown probability density function f0(.), a parametric model*
verbose: Suppose there is a sample x1,x2 ..., xn of n independent and identically distrbuted observations, 
coming from a distribution with an unknown probability density function f0(.)
It is however surmised that the function f0 belongs to a certain family(form in chinese) of distributions {f(.|θ)，θ∈Θ} 
xita, called the parametric model(含参模型) ,  value xita is unknown and is referred to as the the true value of the parameter vector.abs
It is desirable to find an extimator ^xita which would be as close to the true value xita as possible.

To use the method of maximum likelihood, one first specifies the joint density function for all observations. 
*Because the samples is independent :+1:and identically distributed, 
the joint density function can be the multiple multiplication of each density function for each sample.*
f(x1,x2,..,xn|θ)=Πf(xi|θ)  pai θ
#this same function will be called the likelihood:
L(θ;x1,x2,..,xn)=f(x1,x2,..,xn|θ)=Πf(xi|θ)
*Note that ; denotes a separation between the two categories of input arguments: the parameters xita and the observations x1,..,xn*
The method of maximum likelihood estimates θ by finding a value of θ that maximizes l(θ；x)
# ? distribution function <-> model function(map x to y) ?
model function is nothing to do with density distribution

# Find the estimator
For many models, a maximum likelihood estimator can be found as an explicit function of the observed data x1,x2,..,xn.
by derivation: poission / normal 
# system of two element equations
for linear, it is just a problem of linear algebra, for nonlinear ??
For many other models, however, no closed-form solution to the maximization problem is known or available, 
and an MLE has to be found numerically using optimization methods. 
gamma

#program
plot a distribution curve, by the samples
using plt.hist plot a histogram;   get the num and edges of bins, plot the curve
using kde kernel density estimation function, plot the estimated curve
using mlab.normpdf draw the original normal distribution curve




#Exponentiation Terminology
##square:
The expression b2 = b ⋅ b is called the square of b or b squared because the area of a square with side-length b is b2.

##cube:
The expression b3 = b ⋅ b ⋅ b is called the cube of b or b cube the volume of a cube with side-length b is b3.

##xth power  the base 3 appears 5 times in the repeated multiplication
The expression 35 = 3 ⋅ 3 ⋅ 3 ⋅ 3 ⋅ 3 = 243 is call 3 raised to the 5th power. 3 is the base, 5 is the exponent, 243 is the power.
the word "raised" is usually omitted, and sometimes "power" as well, so it can also read 3 to the 5th power,  3 to the 5.


#using sigmoid the hyperbolic tangent function to do normalization:
Normalization is a way of reducing the influence of extreme values of outliers in the data without removing the from the data set.
~ limits the range of the normalized data to values between 0 and 1 or -1 and 1;  
~ is almost linear near the mean 
and has smooth nonliearity at both extremes.