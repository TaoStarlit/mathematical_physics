# coding:utf-8
'''
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
'''

import numpy as np  # so can use mat()  array() directly
import matplotlib.pyplot as plt
from random import *


def Observation(mu=3, sigma=0.1, num=1000):
     # mean and standard deviation
    s = np.random.normal(mu, sigma, num)  # samples
    # add noise
    sr = np.ndarray(np.shape(s))
    i = 0
    for ss in sr:
        d = float(randint(-10, 10)) / 1000
        sr[i] = s[i] + d  # ss=s[i]+d
        # print(s[i],d,sr[i],ss) #(4.9369406581603013, 0.007, 4.1322144706073337e-316, 4.9439406581603009) ss can be read not be write
        i += 1
    return s, sr


def MLE(s):
    mu = np.mean(s)  # dl/dmu =
    sigma = np.var(s)  # dl/dsigma =
    return mu, sigma


from scipy.stats.kde import gaussian_kde


def distrbution_curve(s):
    kde = gaussian_kde(s)
    dist_space = np.linspace(min(s), max(s), 1000)
    return dist_space, kde(dist_space)


def figPlot(name, s, sr, num_bins=50, title1="s", title2="sr"):
    plt.figure(name)

    plt.subplot(1, 2, 1)
    plt.title(title1)
    n, bins, patches = plt.hist(
        s, num_bins, normed=True, facecolor='green', alpha=0.5)
    # print('n: ',n)
    # print(bins,patches)
    # print(len(n),len(bins))#50 51   len(bins[:-1] 50
    half_bin = (bins[1] - bins[0]) / 2
    plt.plot(bins[:-1] + half_bin, n, 'r:')
    x, y = distrbution_curve(s)
    plt.plot(x, y, 'b-')

    plt.subplot(1, 2, 2)
    plt.title(title2)
    n, bins, patches = plt.hist(
        sr, num_bins, normed=True, facecolor='green', alpha=0.5)
    half_bin = (bins[1] - bins[0]) / 2
    plt.plot(bins[:-1] + half_bin, n, 'r-.')
    x, y = distrbution_curve(s)
    plt.plot(x, y, 'b-')


def Main():
    # x,y,xr,yr = loadData()
    # figPlot("figure1 origin and noisy one",x,y,xr,yr,"origin","* 0.8~1.2")
    # print("type(x)",type(x))
    # print("type(xr)",type(xr))
    # X,Y = XY(x,y,9)  #为啥不用 xr yr(有噪声) 来乘方（线性方程组）来算B，却用xr ,yr比较
    # XT=X.transpose()#X的转置
    # B=dot(dot(linalg.inv(dot(XT,X)),XT),Y)#套用最小二乘法公式  , 求出系数B
    # myY=dot(X,B)  #带入回去求重新估算的Y
    # print("shape(x):",shape(x))
    # print("shape(B):",shape(B))
    # figPlot("figure2 model trained by origin",x,myY,x,y,"orginal trained model","origin")

    # #原始生成的数据加上噪声再来训练模型
    # Xr,Yr = XY(xr,yr,9)
    # XTr=Xr.transpose()
    # Br=dot(dot(linalg.inv(dot(XTr,Xr)),XTr),Yr)
    # myYr=dot(X,Br)  #带入回去原始求重新估算的Y
    # figPlot("figure3 model trained by noisy one and origin data",x,myYr,x,y,"model trained by noisy one","origin")
    # figPlot("figure4 model trained by noisy one and noisy data",x,myYr,xr,yr,"model trained by noisy one","origin with noise")
    # plt.show()
    mu, sigma = 2, 0.1
    s, sr = Observation(mu, sigma, 1000)
    print("with noise", MLE(sr))
    print("no noise", MLE(s))
    figPlot("density distribution", s, sr, 100, "no noise", "with noise")

    import matplotlib.mlab as mlab
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma), 'm-')

    plt.show()


Main()
