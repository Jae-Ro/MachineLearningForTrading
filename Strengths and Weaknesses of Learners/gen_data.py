"""
template for generating data to fool learners (c) 2016 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: jro32
GT ID: 903450684
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees


def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    x1 = np.random.random(size=(100, ))*10
    x2 = np.random.random(size=(100, ))*20
    x3 = np.random.random(size=(100, ))*30
    X = np.vstack((x1, x2, x3)).T

    Y = X[:, 0]*3 + 2*X[:, 1] + X[:, 2]*4

    return X, Y


def best4DT(seed=1489683273):
    np.random.seed(seed)
    x1 = np.random.random(size=(100, ))*10
    x2 = np.random.random(size=(100, ))*20
    x3 = np.random.random(size=(100, ))*30
    X = np.vstack((x1, x2, x3)).T
    Y = np.ones((100))
    Y[:50]*0

    return X, Y


def author():
    return 'jro32'  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Tim.")

    for i in range(1000):
        seed = i
        best4LinReg(i)
