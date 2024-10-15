"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 2): KNN
def Q1():
    n_neighbors = [1, 5, 50, 100, 500]
    X,y = make_dataset(3000, random_state=100)
    for neighbors in n_neighbors:
        clf =  KNeighborsClassifier(n_neighbors=neighbors)
        fit = clf.fit(X[:1000], y[:1000])
        plot_boundary("knn_%d" % neighbors, fit, X[1000:], y[1000:], mesh_step_size=0.1, title="n_neighbors = %d" % neighbors)

def Q2():
    accuracies = np.empty((5,5))
    std_deviations = np.empty((1,5))
    n_neighbors = [1, 5, 50, 100, 500]
    seeds = [97, 82, 42, 16, 76]
    for generation in range(5):
        X,y = make_dataset(3000, random_state=seeds[generation])
        for i, neighbors in enumerate(n_neighbors) :
            clf =  KNeighborsClassifier(n_neighbors=neighbors)
            fit = clf.fit(X[:1000], y[:1000])
            y_prediction = fit.predict(X[1000:])
            accuracies[generation, i] = accuracy_score(y[1000:], y_prediction)
    average = accuracies.mean(axis=0)
    std_deviations = accuracies.std(axis=0)
    print("average accuracy =", average) 
    print("standard deviation =", std_deviations)    

if __name__ == "__main__":
    Q1()
    Q2()
