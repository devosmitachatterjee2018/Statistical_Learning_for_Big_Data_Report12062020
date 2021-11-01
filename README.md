## Context
The assignment is a part of the course 'Statistical_Learning_for_Big_Data', course code MVE441 at Chalmers.

## Project 1
A high-dimensional dataset with binary labels is being provided containing 
- the training feature matrix (X_train) of order 323 * 800
- the training binary response vector (y_train) of order 323 * 1
- the validation feature matrix (X_valid) of order 175 * 800
- the validation binary response vector (y_valid) of order 175 * 1.

The project is to analyse the data with two different classification methods on the training dataset, compare the methods’ performance on the validation dataset, determine the best predictors for classification, and explain the selection.

## Responsibilities for project 1
- Perform an exploratory data analysis in order to understand the dataset by summarizing their main characteristics, either statistically or visually.
  *  Data size
  *  Data type
  *  Missing data
  *  Duplicate data
  *  Constant columns
  *  Distribution and count of class labels of the binary response variable
- Standardize the data.
- Since the training response variable (y_train) is binary and the number of features is greater than the number of observations (p > n) in the dataset which motivates to choose the following two penalized logistic regression methods for classification and feature selection.
  * L1–regulated logistic regression
  * Elastic net–regulated logistic regression.

## Project 2
A high-dimensional dataset is being provided in form of the data matrix (X) of order 302 * 728.

The project is to perform an exploratory data analysis, discover clusters in the data, and find five variables that are most indicative of each found cluster.

## Responsibilities for project 2
- Perform an exploratory data analysis in order to understand the dataset by summarizing their main characteristics, either statistically or visually.
  *  Data size
  *  Data type
  *  Missing data
  *  Duplicate data
  *  Constant columns
- Normalize the data.
- Choose t-Distributed Stochastic Neighbor Embedding (tSNE) for dimensionality reduction of the data (X).
- Use k-means clustering on the tSNE reduced data with optimal number of clusters k.

## Environment
Windows, Python.
