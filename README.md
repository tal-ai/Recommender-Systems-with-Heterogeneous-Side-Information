# HIRE_0f34z57i8u

This is a tiny demo code for HIRE on movielens-100k dataset.

## Requirements

- Python 3
- Scikit-learn
- Numpy
- Pandas

## Training

In data file, training data has been splited with names u1.base, u2.base, u3.base, u4.base, u5.base.
Hierarchy matrix and flat feature matrix are available in .txt form in data folder.

All you need is to run train.py in terminal.

## Evaluating
Test data has been defined with names u1.test, u2.test, u3.test, u4.test, u5.test in data folder.

The code will print RMSE value for test data with five fold cross-validation when you run train.py.

## Dataset
The dataset is a copy of the MovieLens | GroupLens 
dataset in the `MovieLens 100k | GroupLens <http://files.grouplens.org/datasets/movielens/ml-100k.zip/>`_
