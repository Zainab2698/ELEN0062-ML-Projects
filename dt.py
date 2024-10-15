"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# (Question 1): Decision Trees

# Put your functions here
def decision_tree(total_points, training_points, depths):
    #Here the total points of DS sample total_points = 3000 & the number of training points is 1000
    iterations = 5
    accuracies = np.empty((iterations, len(depths)))
    random_state_seeds = [97, 82, 42, 16, 76]
    for iteration in range(iterations):
        # Define the dataset
        x, y = make_dataset(n_points=total_points, random_state=random_state_seeds[iteration])

        for i, depth in enumerate(depths):
            tree_clf = DecisionTreeClassifier(max_depth=depth)
            tree_clf.fit(x[:training_points], y[:training_points])

            y_predict = tree_clf.predict(x[training_points:])

            accuracies[iteration, i] = accuracy_score(y[training_points:], y_predict)

    # To store results of the avg accuracy and the std deviation
    average_accuracies = accuracies.mean(axis=0)
    std_devs = accuracies.std(axis=0)

    # Hyperparameter value to test (max_depth)
    # Loop over each max_depth value
    for i, depth in enumerate(depths):
        print(f"Max Depth: {depth} | Avg Accuracy: {average_accuracies[i]:.4f} | Std Dev: {std_devs[i]:.4f}")

        
        if depth is None:
            fname = "decision_boundary_depth_None"  # No formatting needed here
        else:
            fname = f"decision_boundary_depth_{depth}"  # Correctly using f-string

        plot_boundary(fname, fitted_estimator=tree_clf, X=x[training_points:], y=y[training_points:], title="Decision Tree Boundary")


if __name__ == "__main__":
    decision_tree(3000, 1000, depths=[1, 2, 4, 8, None])
