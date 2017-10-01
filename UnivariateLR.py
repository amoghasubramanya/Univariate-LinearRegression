# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:07:10 2017

@author: Amogha Subramanya
Linear regression with one variable to predict profits for a food truck.
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new
outlet. The chain already has trucks in various cities and you have data for
profits and populations from the cities.
3
You would like to use this data to help you select which city to expand
to next.
The file ex1data1.txt contains the dataset for our linear regression problem. 
The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a
loss.
"""
import csv
import matplotlib.pyplot as plt
import math
import numpy as np


def computeCost(X,y,theta):
    J=0
    hx=list()
    m=len(y)
    sqrerrors=list()
    for i in X:
        hofx=theta[0]*i[0]+theta[1]*i[1]              #h(x)=theta0+theta1*x
        hx.append(hofx)
    for i in range(len(y)):
        sqr=hx[i]-y[i]
        sqr=sqr*sqr
        sqrerrors.append(sqr)
    J=1/(2*m)*sum(sqrerrors)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(y)
    J_history=list()
    J_history.append(computeCost(X,y,theta))        #Initial Cost
    for i in range(iterations):
        htheta=list()
        temp=list()
        tempsum=0
        for i in X:
            hofx=theta[0]*1+theta[1]*i[1]              #h(x)=theta0+theta1*x
            htheta.append(hofx)
        for i in range(len(y)):
            err=htheta[i]-y[i]                  #predicted-actual
            temp.append(err)
        theta0=theta[0]-((alpha/m)* sum(temp))
        for i in range(len(X)):
            tempsum+=(temp[i]*X[i][1])
        temp2=(alpha/m)*tempsum
        theta1=theta[1]-temp2
        theta=[theta0,theta1]
        J_history.append(computeCost(X,y,theta))
    return theta,J_history
        


print('Linear regression with one variable to predict profits for a food truck.')  
#Read Data from file
dataset=list()

fp=open('ex1data1.txt','r')
reader = csv.reader(fp, delimiter=',')
for row in reader:
    dataset.append(row)


m=len(dataset)
print('Number of Training examples= ',m)
#Add x0=1 for X matrix
X=list()
#Add x values 
xval=list()
for i in range(m):
    X.append([1])

y=list()
t=0
for i in dataset:
    X[t].append(float(i[0]))
    xval.append(float(i[0]))
    y.append(float(i[1]))
    t+=1

    
#Plotting initial values
def convert(val):
    val=float(val)
    val=math.modf(val)
    return val[1]
xmin=min(xval,key=float)
xmin=convert(xmin)-1

xmax=max(xval,key=float)
xmax=convert(xmax)+1

ymin=min(y,key=float)
ymin=convert(ymin)-1

ymax=max(y,key=float)
ymax=convert(ymax)+1

#Plotting the initial dataset
plotx=[]
ploty=[]
plt.figure(1)
plt.axis([xmin, xmax, ymin, ymax])
#Displaying cluster1

for i in dataset:
	plotx.append(i[0])
	ploty.append(i[1])
plt.plot(plotx, ploty, 'rx')


#Function to plot the regression line after obtaining theta values
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
plt.show()




iterations=1500
alpha=0.01
#Compute and display initial cost (With Theta=[0,0])    
theta=[0,0]
J=computeCost(X,y,theta)
    
print('With Theta=[0,0]')
print('Cost=', J)

print('Gradient Descent:')
print('Number of iterations= ', iterations)
print('Alpha= ',alpha)

#Run GradientDescent
theta,cost = gradientDescent(X, y, theta, alpha, iterations);
print('Theta found by running gradient Descent', theta)

#To predict value for X=[1,3.5] and [1,7]
predict1=[1,3.5]
profit1=predict1[0]*theta[0]+predict1[1]*theta[1]
predict2=[1,7]
profit2=predict2[0]*theta[0]+predict2[1]*theta[1]
abline(theta[1],theta[0])

print(predict1, ' (Profits in areas of 35,000 people)=', profit1)
print(predict2,'(Profits in areas of 70,000 people)=', profit2)


#To Plot Cost versus Number of Iterations
plt.figure(2)
plotx=list()
ploty=list()
plt.axis([-100, len(cost), 0, max(cost)])
for i in range(len(cost)):
    plotx.append(i)
for i in cost:
    ploty.append(i)
plt.plot(plotx, ploty, 'b')
plt.show()