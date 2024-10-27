"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from data import make_dataset
from plot import plot_boundary
from perceptron import PerceptronClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

def knn(irrelevant):
    n = 1
    max_n = 0
    max_score = 0
    X,y = make_dataset(3000, random_state=100, n_irrelevant=irrelevant)
    while n < 500:
        clf =  KNeighborsClassifier(n_neighbors=n)
        score = cross_val_score(clf, X[:1000], y[:1000], cv=5).mean()
        if(score > max_score):
            max_score = score
            max_n = n
        n += 1

    accuracies = np.empty((5,))
    seeds = [97, 82, 42, 16, 76]
    for generation in range(5):
        X,y = make_dataset(3000, random_state=seeds[generation])
        clf =  KNeighborsClassifier(n_neighbors=max_n)
        fit = clf.fit(X[:1000], y[:1000])
        y_prediction = fit.predict(X[1000:])
        accuracies[generation] = accuracy_score(y[1000:], y_prediction)
    average = accuracies.mean()
    std = accuracies.std()
    print("\nTuned parameter = %d with %d irrelevant data" % (max_n, irrelevant))
    print("average accuracy = ", average)
    print("standard deviation = ", std)


def dt(irrelevant):
    max_depth_values = [1, 2, 4, 8, None]
    best_depth_val = None
    best_score = 0

    #Dataset with the n_irrelevant feature
    X, y = make_dataset(3000, random_state=100, n_irrelevant=irrelevant)

    #cross-validation
    for depth in max_depth_values:
        clf = DecisionTreeClassifier(max_depth=depth)

        #Performing the cross-validation with k=5 folds
        cv_scores = cross_val_score(clf, X[:1000], y[:1000], cv=5)
        mean_score = cv_scores.mean()

        #Best max_depth based on cross-validation
        if mean_score > best_score:
            best_score = mean_score
            best_depth_val = depth

    #Model evaluation with regards to the max_depth value
    accuracies = np.empty(5)
    seeds = [97, 82, 42, 16, 76]

    for i, seed in enumerate(seeds):
        X, y = make_dataset(3000, random_state=seed, n_irrelevant=irrelevant)
        clf = DecisionTreeClassifier(max_depth=best_depth_val)
        clf.fit(X[:1000], y[:1000])
        y_prediction = clf.predict(X[1000:])
        accuracies[i] = accuracy_score(y[1000:], y_prediction)

    #Results printing
    avg_accuracy = accuracies.mean()
    std_deviation = accuracies.std()

    print(f"\nTuned max_depth = {best_depth_val} with {irrelevant} irrelevant features")
    print("Average accuracy =", avg_accuracy)
    print("Standard deviation =", std_deviation)


def perceptron(irrelevant):
    etaValues = [10**(-4), 5*10**(-4), 10**(-3), 10**(-2), 10**(-1)]
    bestEta = None
    bestScore = 0

    #Dataset with the n_irrelevant feature
    X, y = make_dataset(3000, random_state=100, n_irrelevant=irrelevant)

    #cross-validation
    for eta in etaValues:
        clf = PerceptronClassifier(learning_rate=eta)

        #Performing the cross-validation with k=5 folds
        cvScores = cross_val_score(clf, X[:1000], y[:1000], cv=5)
        meanScore = cvScores.mean()

        #Best max_depth based on cross-validation
        if meanScore > bestScore:
            bestScore = meanScore
            bestEta = eta

    accuracies = np.empty((5,))
    seeds = [97, 82, 42, 16, 76]
    for generation in range(5):
        X,y = make_dataset(3000, random_state=seeds[generation])
        clf = PerceptronClassifier(learning_rate=bestEta)
        fit = clf.fit(X[:1000], y[:1000])
        y_prediction = fit.predict(X[1000:])
        accuracies[generation] = accuracy_score(y[1000:], y_prediction)
    average = accuracies.mean()
    std = accuracies.std()
    print("\nTuned parameter = ", bestEta," with ", irrelevant," irrelevant data")
    print("average accuracy = ", average)
    print("standard deviation = ", std)


if __name__ == "__main__":
    print("Decisson tree:")
    dt(irrelevant=0)
    dt(irrelevant=200)
    print("\nkNN:")
    knn(irrelevant=0)
    knn(irrelevant=200)
    print("\nPerceptron:")
    perceptron(irrelevant=0)
    perceptron(irrelevant=200)
