#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.metrics import mean_squared_error
import numpy as np

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# you can add other metric here



