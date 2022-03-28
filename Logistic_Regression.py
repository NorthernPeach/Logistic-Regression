#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import math
import numba as nb

def uni_gaus(m, s):
    #Central Limit Theorem
    X = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return X * s**0.5 + m

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ------------------- Grad descent -------------------

@nb.jit
def GD():
    count = 0
    print('Gradient descent:\n')
    W = np.random.rand(m+1, 1) # initial weights
    while True:
        count += 1
        ext = np.dot(X, W) 
        Y = sigmoid(ext)
        derivative = X.T.dot(T - Y)
        W_old = W
        W = W + derivative
        diff = np.linalg.norm(W - W_old)
        if (diff < 1e-3) or (count> 1e4 and diff < 3) or (count > 1e5):
            break
    return W

# ------------------- Newton's method -------------------

@nb.jit
def Newton():
    print("Newton's method:\n")
    W = np.random.rand(m+1, 1)
    count = 0
    Y = np.zeros((N*2, 1))
    D = np.zeros((N*2, N*2))
    while True:
        W_old = W
        count += 1
        for i in range(N*2):
            if math.isinf(np.exp(- np.dot(X[i], W))):
                ext = 100
            else:
                ext = np.dot(X[i], W)
            Y[i] = sigmoid(ext)
            D[i][i] = Y [i] * (1 - Y[i])

        hessian = np.dot(X.T, np.dot(D, X))
        derivative = np.dot(X.T, (T - Y))
        
        if np.linalg.det(hessian) == 0:
            W = W + derivative
        else:
            W = W + np.dot(np.linalg.inv(hessian), derivative)
        diff = np.linalg.norm(W - W_old)
        if (diff < 1e-3) or (count > 1e4 and diff < 3) or (count > 1e5):
            break
    return W

# Visualization

def print_result(data, tpr, tnr, W):
    print('w:\n\t{0}\n\t{1}\n\t{2}'.format(W[0,0], W[1,0], W[2,0]))
    print('\nConfusion matrix:')
    upper_labels = ['Predict cluster 1 ', 'Predict cluster 2']
    bottom_labels = ['Is cluster 1', 'Is cluster 2']
    format_row = "{:>12}" * (len(upper_labels) + 1)
    print(format_row.format("", *upper_labels))
    for cluster, row in zip(bottom_labels, data):
        print(format_row.format(cluster, *row))
    print('')
    print('Sensitivity (Successfully predict cluster 1): ', tpr)
    print('Specificity (Successfully predict cluster 2): ', tnr)
    print('\n-----------------------------------------------')

def output(X, W):
    c1 = []
    c2 = []
    predict = 0
    true_pos = false_pos = true_neg = false_neg = 0
    for i in range(len(X)):
        predict = sigmoid(np.dot(X[i], W))
        if predict >= 0.5:
            c2.append(X[i, 1:])
            if T[i, 0] == 1:
                true_pos += 1
            else:
                false_pos += 1
        else:
            c1.append(X[i, 1:])
            if T[i, 0] == 0:
                true_neg += 1
            else:
                false_neg += 1
    data = [[true_pos, false_pos], [false_neg, true_neg]]
    tpr = true_pos / (true_pos + false_neg)
    tnr = true_neg / (true_neg + false_pos)
    return np.array(c1), np.array(c2), data, tpr, tnr

def Logistic_Regression():

    # plot ground truth
    plt.subplot(131)
    plt.title('Ground Truth')
    plt.scatter(D1[:, 0], D1[:, 1], c='r')
    plt.scatter(D2[:, 0], D2[:, 1], c='b')    

    # Gradient Descent
    W_gd = GD()
    c1, c2, data, tpr, tnr =  output(X, W_gd)
    print_result(data, tpr, tnr, W_gd)
    
    plt.subplot(132)
    plt.title('Gradient Descent')
    if len(c1) != 0:
        plt.scatter(c1[:, 0], c1[:, 1], c='r')
    if len(c2) != 0:
        plt.scatter(c2[:, 0], c2[:, 1], c='b') 

    # Newton's
    W_newton = Newton()
    c1, c2, data, tpr, tnr = output(X, W_newton)
    print_result(data, tpr, tnr, W_newton)
    
    plt.subplot(133)
    plt.title("Newton's Method")
    if len(c1) != 0:
        plt.scatter(c1[:, 0], c1[:, 1], c='r')
    if len(c2) != 0:
        plt.scatter(c2[:, 0], c2[:, 1], c='b') 

    plt.tight_layout()
    plt.show()

# Main function

# input 
N = 50
mx1 = my1 = 1
mx2 = my2 = 3
vx1 = vy1 = 2
vx2 = vy2 = 4
m = 2
# cluster generating
D1 = np.zeros((N, 2))
D2 = np.zeros((N, 2))
for i in range(N):
    D1[i] = [uni_gaus(mx1, vx1), uni_gaus(my1, vy1)] 
    D2[i] = [uni_gaus(mx2, vx2), uni_gaus(my2, vy2)]

X = np.vstack ((D1, D2))
X = np.hstack((np.ones((N*2, 1)), X))
T = np.vstack((np.zeros((N,1)), np.ones((N, 1))))

Logistic_Regression()


