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

    # Define the dataset
    x, y = make_dataset(n_points=total_points)

    x_training = x[:training_points]
    y_training = y[:training_points]
    x_testing = x[training_points:]
    y_testing = y[training_points:]

    

    # To store results of the avg accuracy and the std deviation
    average_accuracies = []
    std_devs = []

    # Hyperparameter value to test (max_depth)
    # Loop over each max_depth value
    for depth in depths:

        # Initialize the classifier with the current max_depth
        tree_clf = DecisionTreeClassifier(max_depth=depth)

        tree_clf.fit(x_training, y_training)

        # Perform 5-fold cross-validation and get accuracy for each fold
        accuracies = cross_val_score(tree_clf, x, y, cv=5, scoring='accuracy')
    
        # Calculate mean and standard deviation of the accuracies
        avg_accuracy = np.mean(accuracies)
        std_dev = np.std(accuracies)
    
        # Store the results
        average_accuracies.append(avg_accuracy)
        std_devs.append(std_dev)

        # Visualize the decision boundary
        if depth is None:
            fname = "decision_boundary_depth_None"  # No formatting needed here
        else:
            fname = f"decision_boundary_depth_{depth}"  # Correctly using f-string

        plot_boundary(fname, fitted_estimator=tree_clf, X=x_testing, y=y_testing, title="Decision Tree Boundary")

    
        # Print the result for the max_depth
        print(f"Max Depth: {depth} | Avg Accuracy: {avg_accuracy:.4f} | Std Dev: {std_dev:.4f}")

    # Printing the Results of Accuracy and Deviation
    print("Average Accuracies:", average_accuracies)
    print("Standard Deviations:", std_devs)


if __name__ == "__main__":
    decision_tree(3000, 1000, depths=[1, 2, 4, 8, None])
