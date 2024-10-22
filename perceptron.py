"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math

from data import make_dataset
from plot import plot_boundary
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 3): Perceptron

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivate(x):
    return sigmoid(x)*(1-sigmoid(x))

def gradient(X, y, w):
    gradient = np.empty(3)
    gradient[0] = (-y/(sigmoid(w[0]+w[1]*X[0]+w[2]*X[1])))*sigmoidDerivate(w[0]+w[1]*X[0]+w[2]*X[1])
    gradient[0] += ((1-y)/(1-sigmoid(w[0]+w[1]*X[0]+w[2]*X[1])))*sigmoidDerivate(w[0]+w[1]*X[0]+w[2]*X[1])
    gradient[1] = gradient[0]*X[0]
    gradient[2] = gradient[0]*X[1]
    return gradient

class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5, learning_rate=.0001):
        self.W = np.random.rand(3)
        self.n_iter = n_iter
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit a perceptron model on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        # Input validation
        X = np.asarray(X, dtype=np.float64)
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        nbEpoch = math.ceil(len(X)/self.n_iter)
        for i in range(nbEpoch):
            temp = 0
            for j in range(self.n_iter):
                temp += gradient(X[self.n_iter*i+j], y[self.n_iter*i+j], self.W)
            self.W = self.W - self.learning_rate * (temp/self.n_iter)
        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        proba = self.predict_proba(X)
        prediction = np.zeros(len(X))
        for i in range(len(X)):
            if proba[i][1] >= 0.5:
                prediction[i] = 1
        return prediction

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        #2 is hard coded for the 2 class we have in this problem
        probabilities = np.zeros((len(X), 2))
        for i in range(len(X)):
            temp = sigmoid(self.W[0] + self.W[1]*X[i][0] + self.W[2]*X[i][1])
            probabilities[i][0] = 1-temp
            probabilities[i][1] = temp
        
        return probabilities


if __name__ == "__main__":
    eta = [10**(-4), 5*10**(-4), 10**(-3), 10**(-2), 10**(-1)]
    X, y = make_dataset(3000, random_state=100)
    for e in eta:
        clf = PerceptronClassifier(learning_rate = e)
        fit = clf.fit(X[:1000], y[:1000])
        plot_boundary("perceptron_learning_rate=%.5f" %e, fit, X[1000:], y[1000:], mesh_step_size=0.1, title="perceptron = %.5f" %e)

    accuracies = np.empty((5, 5))
    stdDeviation = np.empty((5, 5))
    seeds = [97, 82, 42, 16, 76]
    for generation in range(5):
        X, y = make_dataset(3000, random_state=seeds[generation])
        for i, e in enumerate(eta):
            clf = PerceptronClassifier(learning_rate = e)
            fit = clf.fit(X[:1000], y[:1000])
            y_prediction = fit.predict(X[1000:])
            accuracies[generation, i] = accuracy_score(y[1000:], y_prediction)
    average = accuracies.mean(axis=0)
    std_deviation = accuracies.std(axis = 0)
    print("average accuracy =", average)
    print("standard deviation =", std_deviation)
