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
    confusion_matrices = []
    random_state_seeds = [97, 82, 42, 16, 76]
    for iteration in range(iterations):
        # Define the dataset
        x, y = make_dataset(n_points=total_points, random_state=random_state_seeds[iteration])

        for i, depth in enumerate(depths):
            tree_clf = DecisionTreeClassifier(max_depth=depth)
            tree_clf.fit(x[:training_points], y[:training_points])

            y_predict = tree_clf.predict(x[training_points:])

            accuracies[iteration, i] = accuracy_score(y[training_points:], y_predict)
            
            # Confusion matrix
            '''conf_matrix = confusion_matrix(y[training_points:], y_predict)
            confusion_matrices.append(conf_matrix)'''
            if iteration == iterations - 1:
                if depth is None:
                    fname = "decision_boundary_depth_None"  # No formatting needed here
                else:
                    fname = f"decision_boundary_depth_{depth}"  # Correctly using f-string

                plot_boundary(fname, fitted_estimator=tree_clf, X=x[training_points:], y=y[training_points:], title="Decision Tree Boundary")


    # To store results of the avg accuracy and the std deviation
    average_accuracies = accuracies.mean(axis=0)
    std_devs = accuracies.std(axis=0)

    # Hyperparameter value to test (max_depth)
    # Loop over each max_depth value
    for i, depth in enumerate(depths):
        print(f"Max Depth: {depth} | Avg Accuracy: {average_accuracies[i]:.4f} | Std Dev: {std_devs[i]:.4f}")

#Answer to Questions 2 & 3 from the comparison method section
def tuned_dt(irrelevant):
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
    


if __name__ == "__main__":
    decision_tree(3000, 1000, depths=[1, 2, 4, 8, None])
    tuned_dt(irrelevant=0)
    tuned_dt(irrelevant=200)

    '''The optimal hyperparameter setting is usually the one with a good combination
         of high cross-validation mean scores and low standard deviations,
         indicating that it performs well across different folds (datasets).'''
    
