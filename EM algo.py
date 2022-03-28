#!/usr/bin/env python
# coding: utf-8

import os
import gzip
import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb

def get_data(path):
    
    files = os.listdir(path)
    test_img, test_labels, train_img, train_labels = ([gzip.open(path+'{}'.format(file), 'rb').read() for file in files])

    # ------------------------------------Data preprocessing ------------------------------------
    magic, n_img, rows, cols = [int.from_bytes(train_img[i*4:(i+1)*4], byteorder='big') for i in range(0, 4)]
    unsigned_train_img = np.asarray(list(train_img[16:]))
    processed_data = []

    for i in range(0, len(unsigned_train_img), 784):
        a = np.asarray(unsigned_train_img[i:i + 784])
        processed_data.append(a)
    processed_data = np.asarray(processed_data)
    train_labels = np.asarray(list(train_labels[8:]))
    return processed_data, train_labels, n_img, rows * cols

@nb.jit
def ExpStp(p, pi, eta, X):
    for image in range(N):
        for class_ in range(K):
            eta[image, class_] = pi[class_, 0]
            for pixel in range(D):
                if X[image, pixel] == 1:
                    eta[image, class_] *= p[class_, pixel]
                else:
                    eta[image, class_] *= (1 - p[class_, pixel])
        if np.sum(eta[image,:]) != 0:
            eta[image, :] /= np.sum(eta[image,:])
    return eta # responsibility (latent variable)

@nb.jit
def MaxStp(p, pi, eta, X):
    numb = np.sum(eta, axis=0) # summ over all images for each label (cluster weight)
    for class_ in range(K):
        for pixel in range(D):
            p[class_, pixel] = 0
            for image in range(N):
                p[class_, pixel] += X[image, pixel] * eta[image, class_] # mean updating
            try:
                p[class_, pixel] = p[class_, pixel] / numb[class_]
            except:                                    
                p[class_, pixel] = (p[class_, pixel] + 1e-8) / (numb[class_] + 1e-8 * D)
        pi[class_] = (numb[class_] + 1e-8) / (np.sum(numb) + 1e-8 * K) # update class probability
    return p, pi

def print_image(count, p, diff):
    image = np.zeros((K, D))
    image = np.where(p < 0.5, 0, 1) 
    for class_ in range(K):
        print('class {}:'.format(class_))
        for j in range(28):
            for k in range(28):
                print(image[class_][j * 28 + k], end =' ')
            print('')
        print('')

@nb.jit
def predict_relation(X, p, pi, train_labels):
    gtr_pred = np.zeros((10, 10))
    distr = np.zeros(10)
    gtr_pred_relation = np.full((10), -1)
    for image in range(N):
        for class_ in range(K):
            distr[class_] = pi[class_, 0]
            for pixel in range(D):
                if X[image][pixel] == 1:
                    distr[class_] *= p[class_][pixel]
                else:
                    distr[class_] *= (1 - p[class_][pixel])
        predict = np.argmax(distr)
        gtr_pred[predict, train_labels[image]] += 1
    
    for class_ in range(K):
        idx = np.unravel_index(np.argmax(gtr_pred, axis=None), (10, 10))
        gtr_pred_relation[idx[0]] = idx[1]
        for cl in range(K):
            gtr_pred[idx[0]][cl] = -1
            gtr_pred[cl][idx[1]] = -1
    return gtr_pred, gtr_pred_relation

@nb.jit
def print_labeled_image(p, gtr_pred_relation):
    image = np.where(p < 0.5, 0, 1) 
    for cl1 in range(K):
        for cl2 in range(K):
            if gtr_pred_relation[cl2] == cl1:
                label_is = cl2
        print('labeled class {}:'.format(cl1))
        for j in range(28):
            for k in range(28):
                print(image[label_is][j * 28 + k], end =' ')
            print('')
        print('')
    print('-----------------------------------------------------------------------\n')

@nb.jit
def predict_upd(X, p, pi, gtr_pred_relation, train_labels):
    gtr_pred = np.zeros((10, 10))
    distr = np.zeros(10)
    for image in range(N):
        for class_ in range(K):
            distr[class_] = pi[class_, 0]
            for pixel in range(D):
                if X[image][pixel] == 1:
                    distr[class_] *= p[class_][pixel]
                else:
                    distr[class_] *= (1 - p[class_][pixel])
        predict = np.argmax(distr)
        gtr_pred[gtr_pred_relation[predict], train_labels[image]] += 1
    return gtr_pred 

@nb.jit
def confusion_params(class_, gtr_pred):
    tp = fn = fp = tn = 0
    for predict in range(K):
        for target in range(K):
            if predict == class_ and target == class_:
                tp += gtr_pred[predict, target]
            elif predict == class_:
                fp += gtr_pred[predict, target]
            elif target == class_:
                fn += gtr_pred[predict, target]
            else:
                tn += gtr_pred[predict, target]
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    return tp, fn, fp, tn, tpr, tnr 

def conf_mtrx(N, gtr_pred, count):
    error = N
    for class_ in range(K):
        tp, fn, fp, tn, tpr, tnr = confusion_params(class_, gtr_pred)
        error -= tp
        print('Confusion Matrix {}:'.format(class_))
        print('{:^20}{:^25}{:^25}'.format(' ', 'Predict number %d'%class_, 'Predict not number %d'%class_))
        print('{:^20}{:^25}{:^25}'.format('Is number %d'%class_, int(tp), int(fn)))
        print('{:^20}{:^25}{:^25}\n'.format('Isn\'t number %d'%class_, int(fp), int(tn)))
        print('Sensitivity (Successfully predict number {0}):     {1}'.format(class_, tpr))
        print('Specificity (Successfully predict not number {0}): {1}'.format(class_, tnr)) 
        print('\n-----------------------------------------------------------------------\n')
    print('\nTotal iteration to converge:', count)
    print('Total error rate:', error/N)


# ------------------------------------ Main function ------------------------------------
path = 'PATH TO YOUR FOLDER'
processed_data, train_labels, N, D = get_data(path)
K = 10
X = np.zeros((N, D), dtype=int) 
for r in range(N):
    for c in range(D):
        if (processed_data[r, c]>=128):
            X[r, c] = 1
            
# initial params
eta = np.zeros((N, K)) 
p = np.random.uniform(0.0, 1.0, (10,784)) 
for class_ in range(K):
    total = np.sum(p[class_, :])
    p[class_, :] /= total

pi = np.full((10, 1), 0.1) # assume each class equally to happen (uniform prob)

# EM algo
count = 0
while True:
    count += 1
    p_old = p
    eta = ExpStp(p, pi, eta, X) 
    p, pi = MaxStp(p, pi, eta, X) 
    diff = np.linalg.norm(p - p_old)
    print_image(count, p, diff)
    print('No. of Iteration: {}, Difference: {}\n'.format(count, diff))
    print('---------------------------------------------------------------\n')
    if (count >= 30 and diff < 1e-8):
        break
    
gtr_pred, gtr_pred_relation = predict_relation(X, p, pi, train_labels)
print_labeled_image(p, gtr_pred_relation)
gtr_pred = predict_upd(X, p, pi, gtr_pred_relation, train_labels)
conf_mtrx(N, gtr_pred, count)

